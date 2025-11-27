#!/usr/bin/env python3
"""
db_analyzer.py

Polished business-ready SQLite analyzer and automated emailer.

Usage:
    python db_analyzer.py --db data.db --email recipient@example.com --team "Team Name"

SMTP credentials via environment variables:
    SMTP_SERVER, SMTP_PORT, SENDER_EMAIL, SENDER_PASSWORD
"""

import os
import sys
import argparse
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import traceback
from datetime import datetime, timezone
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# --------------- Configuration / Utilities ---------------

sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.autolayout': True})

def safe_mkdir(path):
    os.makedirs(path, exist_ok=True)

def open_conn(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn

def save_fig(fig, path, dpi=300):
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

# --------------- DB Discovery / Read ---------------

def list_tables(conn):
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    return [r[0] for r in cur.fetchall()]

def read_table(conn, table, limit=None):
    try:
        q = f"SELECT * FROM '{table}'"
        if limit:
            q += f" LIMIT {limit}"
        df = pd.read_sql_query(q, conn)
    except Exception:
        # fallback limited sample
        df = pd.read_sql_query(f"SELECT * FROM '{table}' LIMIT 1000", conn)
    return df

def table_columns(conn, table):
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info('{table}')")
    return [(r[1], r[2]) for r in cur.fetchall()]

def analyze_table(df, table_name):
    out = {}
    out['table'] = table_name
    out['rows'] = len(df)
    out['columns'] = list(df.columns)
    out['dtypes'] = df.dtypes.apply(lambda x: str(x)).to_dict()
    out['nulls'] = df.isnull().sum().to_dict()
    out['duplicates'] = int(df.duplicated().sum())
    try:
        out['describe'] = df.describe(include='all').to_dict()
    except Exception:
        out['describe'] = {}
    return out

# --------------- Heuristics for business fields ---------------

def find_datetime_column(df):
    for col in df.columns:
        lname = col.lower()
        if 'date' in lname or 'time' in lname:
            return col
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):
            return col
    return None

def find_customer_column(df):
    # heuristics: name contains 'customer' or 'cust' or 'bill-to'
    for col in df.columns:
        lname = col.lower()
        if 'customer' in lname or 'cust' in lname or 'bill' in lname:
            return col
    return None

def find_item_column(df):
    for col in df.columns:
        lname = col.lower()
        if 'item' in lname or 'product' in lname or 'sku' in lname:
            return col
    return None

def find_numeric_value_columns(df):
    # numeric columns likely to be sales/amount/revenue/total
    candidates = []
    for col in df.select_dtypes(include=[np.number]).columns:
        lname = col.lower()
        if any(k in lname for k in ['sales', 'amount', 'revenue', 'total', 'price', 'amt', 'value']):
            candidates.append(col)
    # fallback include any numeric
    if not candidates:
        candidates = list(df.select_dtypes(include=[np.number]).columns)
    return candidates

# --------------- Charts ---------------

def make_count_chart(all_dfs, output_dir):
    """Bar of top categories or table row counts."""
    candidate = None
    for table_name, df in all_dfs.items():
        # look for categorical columns with reasonable unique counts
        for col in df.columns:
            try:
                nunique = int(df[col].nunique(dropna=True))
            except Exception:
                nunique = 0
            if 1 < nunique <= 40 and df[col].dtype == object:
                candidate = (table_name, col)
                break
        if candidate:
            break

    path = os.path.join(output_dir, 'chart1_count.png')
    if candidate:
        table_name, col = candidate
        ser = all_dfs[table_name][col].value_counts().nlargest(20)
        fig, ax = plt.subplots(figsize=(12,8))
        sns.barplot(x=ser.values, y=ser.index, ax=ax)
        ax.set_title(f'Top categories for {table_name}.{col}', fontsize=14)
        ax.set_xlabel('Count')
        ax.set_ylabel(col)
        save_fig(fig, path)
        return path, f'Count chart of {table_name}.{col}'
    else:
        rows = {t: len(df) for t, df in all_dfs.items()}
        items = sorted(rows.items(), key=lambda x: x[1], reverse=True)[:20]
        names = [i[0] for i in items]
        vals = [i[1] for i in items]
        fig, ax = plt.subplots(figsize=(12,8))
        sns.barplot(x=vals, y=names, ax=ax)
        ax.set_title('Row counts per table', fontsize=14)
        ax.set_xlabel('Rows')
        ax.set_ylabel('Table')
        save_fig(fig, path)
        return path, 'Row counts per table'

def make_trend_chart(all_dfs, output_dir):
    """Time-series counts by month for found date column or fallback numeric trend."""
    path = os.path.join(output_dir, 'chart2_trend.png')
    for table_name, df in all_dfs.items():
        col = find_datetime_column(df)
        if col is not None:
            ser = pd.to_datetime(df[col], errors='coerce')
            ser = ser.dropna()
            if len(ser) < 2:
                continue
            counts = ser.dt.to_period('M').value_counts().sort_index()
            idx = counts.index.to_timestamp()
            fig, ax = plt.subplots(figsize=(12,6))
            ax.plot(idx, counts.values, marker='o')
            ax.set_title(f'Trend over time ({table_name}.{col})', fontsize=14)
            ax.set_xlabel('Date')
            ax.set_ylabel('Count')
            save_fig(fig, path)
            return path, f'Trend chart for {table_name}.{col}'

    # fallback: numeric column trend in largest table
    largest_table = max(all_dfs.items(), key=lambda x: len(x[1]))
    table_name, df = largest_table
    nums = find_numeric_value_columns(df)
    if nums and len(df) >= 3:
        col = nums[0]
        fig, ax = plt.subplots(figsize=(12,6))
        sample = df[col].fillna(0).astype(float)
        ax.plot(sample.index, sample.values, linewidth=1)
        ax.set_title(f'{table_name} - {col} over index (fallback trend)', fontsize=14)
        ax.set_xlabel('Row index')
        ax.set_ylabel(col)
        save_fig(fig, path)
        return path, f'Fallback trend chart for {table_name}.{col}'

    fig, ax = plt.subplots(figsize=(8,4))
    ax.text(0.5, 0.5, 'No trendable fields found', ha='center')
    save_fig(fig, path)
    return path, 'No trendable fields found'

def make_relationship_chart(all_dfs, output_dir):
    """Correlation heatmap or scatter fallback."""
    path = os.path.join(output_dir, 'chart3_relationship.png')

    # Heatmap search
    for table_name, df in all_dfs.items():
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] >= 2 and len(num_df) >= 3:
            corr = num_df.corr()
            fig, ax = plt.subplots(figsize=(14,10))
            sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.3, ax=ax)
            plt.xticks(rotation=45, ha='right', fontsize=8)
            plt.yticks(rotation=0, fontsize=8)
            ax.set_title(f'Correlation heatmap for {table_name} (numeric columns)', fontsize=14)
            save_fig(fig, path)
            return path, f'Correlation heatmap for {table_name}'

    # Scatter fallback
    for table_name, df in all_dfs.items():
        nums = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(nums) >= 2 and len(df) >= 5:
            x, y = nums[0], nums[1]
            fig, ax = plt.subplots(figsize=(10,6))
            ax.scatter(df[x].dropna(), df[y].dropna(), alpha=0.6)
            ax.set_title(f'Scatter: {table_name}.{x} vs {y}', fontsize=12)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
            save_fig(fig, path)
            return path, f'Scatter {table_name}.{x} vs {y}'

    fig, ax = plt.subplots(figsize=(6,4))
    ax.text(0.5, 0.5, 'No numeric relationships found', ha='center')
    save_fig(fig, path)
    return path, 'No numeric relationships found'

# --------------- PDF generation ---------------

def generate_pdf_report(analysis_summary, table_analyses, chart_paths, business_metrics, top_customers, top_items, output_pdf, team_name):
    c = canvas.Canvas(output_pdf, pagesize=A4)
    width, height = A4
    margin = 18 * mm

    # Cover
    c.setFont('Helvetica-Bold', 22)
    c.drawCentredString(width/2, height - 40*mm, f'{team_name} - Database Analysis')
    c.setFont('Helvetica', 11)
    c.drawCentredString(width/2, height - 48*mm, f"Generated: {analysis_summary['analysis_date']}")
    c.drawCentredString(width/2, height - 56*mm, 'AI CODEFIX 2025 - Automated Agent')
    c.showPage()

    # Executive summary & business metrics
    c.setFont('Helvetica-Bold', 16)
    c.drawString(margin, height - margin, 'EXECUTIVE SUMMARY')
    c.setFont('Helvetica', 10)
    y = height - margin - 16
    c.drawString(margin, y, f"Total Tables: {analysis_summary['total_tables']}")
    y -= 12
    c.drawString(margin, y, f"Total Records: {analysis_summary['total_records']:,}")
    y -= 12
    c.drawString(margin, y, f"Analysis Date: {analysis_summary['analysis_date']}")
    y -= 18

    c.setFont('Helvetica-Bold', 12)
    c.drawString(margin, y, "KEY BUSINESS METRICS")
    y -= 14
    c.setFont('Helvetica', 10)
    if business_metrics:
        for k, v in list(business_metrics.items())[:8]:
            if y < 40*mm:
                c.showPage(); y = height - margin
            c.drawString(margin + 8, y, f"- {k}: {v:,.2f}")
            y -= 12
    else:
        c.drawString(margin + 8, y, "No business numeric metrics detected.")
        y -= 12

    y -= 8
    c.setFont('Helvetica-Bold', 12)
    c.drawString(margin, y, "TOP CUSTOMERS")
    y -= 14
    c.setFont('Helvetica', 10)
    if top_customers:
        for cust, amt in top_customers[:8]:
            if y < 40*mm:
                c.showPage(); y = height - margin
            c.drawString(margin + 8, y, f"- {cust}: {amt:,.2f}")
            y -= 12
    else:
        c.drawString(margin + 8, y, "No customer metrics available.")
        y -= 12

    y -= 8
    c.setFont('Helvetica-Bold', 12)
    c.drawString(margin, y, "TOP ITEMS")
    y -= 14
    c.setFont('Helvetica', 10)
    if top_items:
        for it, amt in top_items[:8]:
            if y < 40*mm:
                c.showPage(); y = height - margin
            c.drawString(margin + 8, y, f"- {it}: {amt:,.2f}")
            y -= 12
    else:
        c.drawString(margin + 8, y, "No item metrics available.")
        y -= 12

    c.showPage()

    # Per-table condensed details
    for t in table_analyses:
        c.setFont('Helvetica-Bold', 12)
        c.drawString(margin, height - margin, f"Table: {t['table']}")
        c.setFont('Helvetica', 9)
        y = height - margin - 16
        c.drawString(margin, y, f"Rows: {t['rows']}  |  Columns: {len(t['columns'])}  |  Duplicates: {t.get('duplicates', 0)}")
        y -= 12
        c.drawString(margin, y, "Top null counts (first 10):")
        y -= 12
        nulls = t.get('nulls', {})
        for col, val in list(nulls.items())[:10]:
            if y < 30*mm:
                c.showPage(); y = height - margin
            c.drawString(margin + 8, y, f"{col}: {val}")
            y -= 10
        c.showPage()

    # Visualizations
    c.setFont('Helvetica-Bold', 14)
    c.drawString(margin, height - margin, 'VISUALIZATIONS')
    y = height - margin - 18
    for p in chart_paths:
        try:
            if y < 90*mm:
                c.showPage(); y = height - margin
            img = ImageReader(p['path'])
            iw, ih = img.getSize()
            aspect = ih / iw
            display_w = width - 2*margin
            display_h = display_w * aspect
            if display_h > y - margin:
                display_h = y - margin
                display_w = display_h / aspect
            c.drawImage(img, margin, y - display_h, width=display_w, height=display_h)
            y = y - display_h - 12
            c.setFont('Helvetica', 10)
            c.drawString(margin, y, p.get('desc', ''))
            y -= 18
        except Exception:
            pass

    c.save()

# --------------- Email ---------------

def build_html_email(subject, analysis_summary, business_metrics, insights, chart_paths, team_name):
    # Build a concise, business-ready HTML email
    metrics_html = ""
    for k, v in list(business_metrics.items())[:6]:
        metrics_html += f"<li><strong>{k}</strong>: {v:,.2f}</li>"

    insights_html = ""
    for i, it in enumerate(insights[:6], 1):
        insights_html += f"<li>{i}. {it}</li>"

    attachments_html = ", ".join([os.path.basename(p['path']) for p in chart_paths] + ['report.pdf'])

    html = f"""
    <html>
      <body style="font-family: Arial, sans-serif; color:#222;">
        <h2 style="color:#2b6cb0;">{subject}</h2>
        <p>Dear Recipient,</p>
        <p>Please find attached the automated <strong>business analytics</strong> report produced by <em>{team_name}</em>.</p>

        <h3>DATABASE SUMMARY</h3>
        <ul>
          <li><strong>Total Tables:</strong> {analysis_summary['total_tables']}</li>
          <li><strong>Total Records:</strong> {analysis_summary['total_records']:,}</li>
          <li><strong>Analysis Date:</strong> {analysis_summary['analysis_date']}</li>
        </ul>

        <h3>KEY BUSINESS METRICS</h3>
        <ul>
          {metrics_html if metrics_html else '<li>No numeric business metrics detected.</li>'}
        </ul>

        <h3>KEY INSIGHTS</h3>
        <ul>
          {insights_html if insights_html else '<li>No automated insights generated.</li>'}
        </ul>

        <p><strong>Attachments:</strong> {attachments_html}</p>

        <p>Best regards,<br/><strong>{team_name}</strong><br/>AI CODEFIX 2025</p>

      </body>
    </html>
    """
    return html

def send_email(smtp_server, smtp_port, sender, password, recipient, subject, html_body, attachments):
    msg = MIMEMultipart('alternative')
    msg['From'] = sender
    msg['To'] = recipient
    msg['Subject'] = subject

    # Plain-text fallback
    plain = f"{subject}\n\nPlease see attached report and charts."

    part1 = MIMEText(plain, 'plain')
    part2 = MIMEText(html_body, 'html')
    msg.attach(part1)
    msg.attach(part2)

    # Attach files
    for path in attachments:
        try:
            with open(path, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename="{os.path.basename(path)}"')
            msg.attach(part)
        except Exception as e:
            print(f"Warning: could not attach {path}: {e}")

    # Send
    try:
        server = smtplib.SMTP(smtp_server, int(smtp_port))
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, [recipient], msg.as_string())
        server.quit()
        return True, None
    except Exception as e:
        return False, str(e)

# --------------- Orchestration ---------------

def main(args):
    out_dir = args.output or 'output'
    safe_mkdir(out_dir)

    # Connect DB
    try:
        conn = open_conn(args.db)
    except Exception as e:
        print(f"ERROR: could not open database {args.db}: {e}")
        sys.exit(1)

    tables = list_tables(conn)
    all_dfs = {}
    table_analyses = []
    total_records = 0

    for t in tables:
        try:
            df = read_table(conn, t)
            # try convert common date patterns
            for col in df.columns:
                if df[col].dtype == object:
                    if df[col].astype(str).str.contains(r"\d{4}-\d{2}-\d{2}", na=False).any():
                        try:
                            df[col] = pd.to_datetime(df[col], errors='coerce')
                        except Exception:
                            pass
            all_dfs[t] = df
            ta = analyze_table(df, t)
            table_analyses.append(ta)
            total_records += ta['rows']
        except Exception as e:
            print(f"Warning: failed to read table {t}: {e}")

    # --- Compute basic business metrics (inside main) ---
    business_metrics = {}
    for table, df in all_dfs.items():
        for col in df.columns:
            lname = col.lower()
            if any(k in lname for k in ['sales', 'amount', 'revenue', 'total', 'price', 'amt', 'value']):
                try:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        val = float(df[col].fillna(0).sum())
                        business_metrics[f"{table}.{col}"] = val
                except Exception:
                    pass

    # Attempt top customers & top items heuristics
    top_customers = []
    top_items = []
    try:
        # heuristic: find a table with customer, item, and numeric amount columns
        for table, df in all_dfs.items():
            cust_col = find_customer_column(df)
            item_col = find_item_column(df)
            num_cols = find_numeric_value_columns(df)
            if cust_col and num_cols:
                # aggregate by customer on first numeric column
                amt_col = num_cols[0]
                agg = df[[cust_col, amt_col]].dropna()
                if not agg.empty:
                    grp = agg.groupby(cust_col)[amt_col].sum().sort_values(ascending=False)
                    top_customers = [(str(idx), float(v)) for idx, v in grp.head(10).items()]
            if item_col and num_cols:
                amt_col = num_cols[0]
                agg2 = df[[item_col, amt_col]].dropna()
                if not agg2.empty:
                    grp2 = agg2.groupby(item_col)[amt_col].sum().sort_values(ascending=False)
                    top_items = [(str(idx), float(v)) for idx, v in grp2.head(10).items()]
            # break early if we have data
            if top_customers and top_items:
                break
    except Exception:
        pass

    analysis_summary = {
        'total_tables': len(tables),
        'total_records': total_records,
        'analysis_date': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%SZ'),
        'insights': []
    }

    # Simple automated insights
    if len(tables) == 0:
        analysis_summary['insights'].append('No tables found in the database.')
    else:
        if table_analyses:
            largest = max(table_analyses, key=lambda x: x['rows'])
            analysis_summary['insights'].append(f'Largest table: {largest["table"]} with {largest["rows"]} rows.')
        for t in table_analyses:
            nulls = t.get('nulls', {})
            cols_many_nulls = [c for c, v in nulls.items() if t['rows'] > 0 and (v / max(1, t['rows'])) > 0.5]
            if cols_many_nulls:
                analysis_summary['insights'].append(f'Table {t["table"]} has columns with >50% nulls: {", ".join(cols_many_nulls)}')
        numeric_tables = [t for t, df in all_dfs.items() if df.select_dtypes(include=[np.number]).shape[1] >= 2]
        if numeric_tables:
            analysis_summary['insights'].append(f'Numeric correlations available in tables: {", ".join(numeric_tables[:5])}')

    # Create charts
    chart_paths = []
    try:
        p1, d1 = make_count_chart(all_dfs, out_dir)
        chart_paths.append({'path': p1, 'desc': d1})
    except Exception as e:
        print('Chart1 failed:', e)
    try:
        p2, d2 = make_trend_chart(all_dfs, out_dir)
        chart_paths.append({'path': p2, 'desc': d2})
    except Exception as e:
        print('Chart2 failed:', e)
    try:
        p3, d3 = make_relationship_chart(all_dfs, out_dir)
        chart_paths.append({'path': p3, 'desc': d3})
    except Exception as e:
        print('Chart3 failed:', e)

    # Generate PDF
    pdf_path = os.path.join(out_dir, 'report.pdf')
    try:
        generate_pdf_report(analysis_summary, table_analyses, chart_paths, business_metrics, top_customers, top_items, pdf_path, args.team or "Team")
    except Exception:
        print('PDF generation failed:')
        traceback.print_exc()

    # Prepare email
    smtp_server = args.smtp_server or os.getenv('SMTP_SERVER')
    smtp_port = args.smtp_port or os.getenv('SMTP_PORT', 587)
    sender = args.sender or os.getenv('SENDER_EMAIL')
    sender_pass = os.getenv('SENDER_PASSWORD')

    missing = []
    if not smtp_server: missing.append('SMTP_SERVER')
    if not smtp_port: missing.append('SMTP_PORT')
    if not sender: missing.append('SENDER_EMAIL')
    if not sender_pass: missing.append('SENDER_PASSWORD')
    if missing:
        print('Missing SMTP configuration for:', ', '.join(missing))
        print('Report saved locally at:', pdf_path)
        print('Charts saved in:', out_dir)
        print('Exiting without sending email.')
        return

    subject = f"Database Analysis Report - {args.team or 'Team'}"

    html_body = build_html_email(subject, analysis_summary, business_metrics, analysis_summary.get('insights', []), chart_paths, args.team or "Team")

    attachments = [pdf_path] + [p['path'] for p in chart_paths]

    ok, err = send_email(smtp_server, smtp_port, sender, sender_pass, args.email, subject, html_body, attachments)
    if ok:
        print('Email sent successfully to', args.email)
    else:
        print('Failed to send email:', err)
        print('Report saved locally at:', pdf_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Automated DB Analyzer and Email Reporter')
    parser.add_argument('--db', required=True, help='Path to sqlite database file')
    parser.add_argument('--email', required=True, help='Recipient email')
    parser.add_argument('--output', help='Output directory', default='output')
    parser.add_argument('--smtp-server', help='SMTP server (optional, or set SMTP_SERVER env var)')
    parser.add_argument('--smtp-port', help='SMTP port (optional, or set SMTP_PORT env var)')
    parser.add_argument('--sender', help='Sender email (optional, or set SENDER_EMAIL env var)')
    parser.add_argument('--team', help='Team name to include in subject/body', default='Your Team')

    args = parser.parse_args()
    main(args)
