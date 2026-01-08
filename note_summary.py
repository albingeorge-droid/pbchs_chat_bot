from langsmith import traceable

from db import run_select
from openai_client import GroqClient
from prompts import NOTE_SUMMARY_SYSTEM_PROMPT

try:
    # pip install fpdf2
    from fpdf import FPDF

    class BorderedPDF(FPDF):
        def header(self):
            """
            Draw a border inside the page margins on *every* page.
            This is called automatically whenever a page is created
            (including auto page breaks).
            """

            BORDER_MARGIN = 5
            # outer border
            self.set_line_width(0.3)
            self.rect(
                BORDER_MARGIN,                      # left
                BORDER_MARGIN,                      # top
                self.w - 2 * BORDER_MARGIN,         # width
                self.h - 2 * BORDER_MARGIN,         # height
            )
            # reset cursor to top-left inside the border
            self.set_xy(self.l_margin, self.t_margin)

except ImportError:
    FPDF = None
    BorderedPDF = None


from typing import List, Dict, Tuple
import os


@traceable(run_type="chain", name="generate_property_note_pdf")
def generate_property_note_pdf(
    llm: GroqClient,
    pra: str,
    output_dir: str = "property_notes",
) -> Tuple[str, str, List[Dict[str, any]], List[Dict[str, any]]]:
    """
    Generate a note summary PDF for a single property PRA.

    Returns:
      (summary_text, pdf_path, current_owner_rows, history_rows)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Basic safety: escape single quotes for literal
    safe_pra = pra.replace("'", "''")

    # 1) Current owners
    sql_current = f"""
    SELECT T2.name AS owner_name, T1.buyer_portion
    FROM current_owners AS T1
    JOIN persons AS T2 ON T1.buyer_id = T2.id
    JOIN properties AS T3 ON T1.property_id = T3.id
    WHERE T3.pra = '{safe_pra}'
    LIMIT 50;
    """.strip()

    # 2) Ownership history
    sql_history = f"""
    SELECT T3.name AS buyer_name,
           T5.name AS seller_name, 
           (T4.signing_date->>0) AS signing_date,
           T2.buyer_portion,
           T2.transfer_type,
           T2.notes
    FROM properties AS T1
    JOIN ownership_records AS T2 ON T1.id = T2.property_id
    JOIN persons AS T3 ON T2.buyer_id = T3.id
    JOIN sale_deeds AS T4 ON T2.sale_deed_id = T4.id
    JOIN ownership_sellers AS T6 ON T6.ownership_id = T2.id
    JOIN persons AS T5 ON T5.id = T6.person_id
    WHERE T1.pra = '{safe_pra}'
    LIMIT 50;
    """.strip()

    current_rows = run_select(sql_current)   # uses your guardrails 
    history_rows = run_select(sql_history)

    # 3) Initial plot size
    sql_initial_size = f"""
    SELECT T2.initial_plot_size
    FROM properties AS T1
    JOIN property_addresses AS T2 ON T1.id = T2.property_id
    WHERE T1.pra = '{safe_pra}'
    LIMIT 1;
    """.strip()
    initial_size_rows = run_select(sql_initial_size) or []
    initial_plot_size = (
        initial_size_rows[0].get("initial_plot_size")
        if initial_size_rows
        else None
    )

    # 4) Share certificate details
    sql_share = f"""
    SELECT 
           T2.certificate_number,
           T2.date_of_transfer,
           T3.name AS member_name
    FROM properties AS T1
    JOIN share_certificates AS T2 ON T1.id = T2.property_id
    JOIN persons AS T3 ON T2.member_id = T3.id
    WHERE T1.pra = '{safe_pra}'
    LIMIT 50;
    """.strip()
    share_rows = run_select(sql_share) or []


    # 5) Club membership details
    sql_club = f"""
    SELECT 
           T1.membership_number,
           T2.name AS member_name,
           T1.allocation_date
    FROM club_memberships AS T1
    JOIN persons AS T2 ON T1.member_id = T2.id
    JOIN properties AS T3 ON T1.property_id = T3.id
    WHERE T3.pra = '{safe_pra}'
    LIMIT 50;
    """.strip()
    club_rows = run_select(sql_club) or []


    # --- Build PDF in tabular format like the sample ---
    if BorderedPDF is None:
        raise RuntimeError("FPDF is not installed; cannot generate PDF")

    pdf = BorderedPDF()
    pdf.set_auto_page_break(auto=True, margin=10)
    pdf.add_page()
    # ❌ no manual outer border here – header() in BorderedPDF draws it on every page

    # small padding between outer border and all tables
    inner_margin = 3  # you can use 4–5 if you want a bigger gap

    # Compute usable width for tables *inside* the border padding
    page_width = pdf.w - pdf.l_margin - pdf.r_margin - 2 * inner_margin


    # Title: 28|6|Punjabi Bagh East (centered, underlined)
    # ----- Pretty title from PRA -----
    # PRA format is "plot|road|Punjabi Bagh East/West"
    parts = [p.strip() for p in pra.split("|")]
    if len(parts) == 3:
        plot_no, road_no, area = parts

        # Optional: turn "Punjabi Bagh East" -> "East Punjabi Bagh"
        area_words = area.split()
        if (
            len(area_words) == 3
            and area_words[0].lower() == "punjabi"
            and area_words[1].lower() == "bagh"
        ):
            side = area_words[2]           # "East" / "West"
            pretty_area = f"{side} Punjabi Bagh"
        else:
            pretty_area = area

        title_text = f"Plot no. {plot_no} Road no. {road_no} {pretty_area}"
    else:
        # Fallback: just use PRA as-is
        title_text = pra

    # Title (centered, underlined)
    pdf.set_font("Arial", "BU", 14)
    pdf.cell(0, 10, title_text, ln=1, align="C")


    pdf.ln(5)

    # ✅ Initial plot size line
    if initial_plot_size:
        pdf.set_font("Arial", size=11)
        pdf.cell(
            0,
            8,
            f"Initial plot size of the property: {initial_plot_size} sq. yards",
            ln=1,
        )
        pdf.ln(3)

    # Subtitle line
    pdf.set_font("Arial", size=11)
    pdf.cell(
        0,
        8,
        "All the transaction that has happened is listed below:",
        ln=1,
    )

    pdf.ln(3)


    # ---------- Transactions table ----------
    # Columns: S.No. | Date | Buyer | Seller | Portion | Transfer_type | Note
    col_widths = [
        10,   # S.No.
        20,   # Date
        38,   # Buyer
        38,   # Seller
        18,   # Portion
        25,   # Transfer_type
        page_width - (10 + 20 + 38 + 38 + 18 + 25),  # Note
    ]

    headers = ["S.No.", "Date", "Buyer", "Seller", "Portion", "Transfer Type", "Note"]

    pdf.set_font("Arial", "B", 10)
    pdf.set_x(pdf.l_margin + inner_margin)
    for w, header in zip(col_widths, headers):
        pdf.cell(w, 8, header, border=1, align="C")
    pdf.ln(8)

    pdf.set_font("Arial", "", 9)
    line_height = 6  # height per text line inside a row

    def _clean(text: str) -> str:
        """Sanitize text so FPDF is happy (latin-1, no control chars)."""
        text = (text or "").replace("\t", " ").replace("\n", " ")
        text = text.encode("latin-1", "replace").decode("latin-1")
        return "".join(ch if ord(ch) >= 32 else " " for ch in text)

    def wrap_to_width(pdf_obj: FPDF, text: str, max_width: float) -> List[str]:
        """Word-wrap a string so each line fits inside max_width for the current font."""
        words = _clean(text).split()
        if not words:
            return [""]

        lines: List[str] = []
        current = words[0]
        for word in words[1:]:
            test = current + " " + word
            if pdf_obj.get_string_width(test) <= max_width:
                current = test
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
        return lines or [""]

    # Transactions rows
    for idx, row in enumerate(history_rows, start=1):
        serial = f"{idx}."
        date = (row.get("signing_date") or "")[:10]
        buyer = _clean(row.get("buyer_name") or "")
        seller = _clean(row.get("seller_name") or "")
        portion = (
            f'{row.get("buyer_portion"):.2f}'
            if isinstance(row.get("buyer_portion"), (int, float))
            else str(row.get("buyer_portion") or "")
        )
        transfer_type = _clean(row.get("transfer_type") or "")
        note_text = row.get("notes") or ""

        serial_lines = [serial]
        date_lines = [date]
        buyer_lines = wrap_to_width(pdf, buyer, col_widths[2] - 2)
        seller_lines = wrap_to_width(pdf, seller, col_widths[3] - 2)
        portion_lines = [portion]
        transfer_lines = wrap_to_width(pdf, transfer_type, col_widths[5] - 2)
        note_lines = wrap_to_width(pdf, note_text, col_widths[6] - 2)

        num_lines = max(
            len(serial_lines),
            len(date_lines),
            len(buyer_lines),
            len(seller_lines),
            len(portion_lines),
            len(transfer_lines),
            len(note_lines),
        )
        row_height = num_lines * line_height

        # 🔴 manual page-break: if this row won't fit, go to a new page and reprint header
        bottom_limit = pdf.h - pdf.b_margin - inner_margin
        if pdf.get_y() + row_height > bottom_limit:
            pdf.add_page()  # BorderedPDF.header() will redraw the outer border

            # re-draw the transactions table header on the new page
            pdf.set_font("Arial", "B", 10)
            pdf.set_x(pdf.l_margin + inner_margin)
            for w, header in zip(col_widths, headers):
                pdf.cell(w, 8, header, border=1, align="C")
            pdf.ln(8)
            pdf.set_font("Arial", "", 9)

        # now draw this row
        pdf.set_x(pdf.l_margin + inner_margin)
        x_start = pdf.get_x()
        y_start = pdf.get_y()


        # draw borders
        x = x_start
        for w in col_widths:
            pdf.rect(x, y_start, w, row_height)
            x += w

        # fill text
        def draw_column(col_index: int, lines: List[str], align: str = "L"):
            x_col = x_start + sum(col_widths[:col_index])
            for i in range(num_lines):
                text_line = _clean(lines[i]) if i < len(lines) else ""
                y_line = y_start + 1 + i * line_height
                pdf.set_xy(x_col + 1, y_line)
                pdf.cell(
                    col_widths[col_index] - 2,
                    line_height,
                    text_line,
                    border=0,
                    ln=0,
                    align=align,
                )

        draw_column(0, serial_lines, align="L")
        draw_column(1, date_lines, align="L")
        draw_column(2, buyer_lines, align="L")
        draw_column(3, seller_lines, align="L")
        draw_column(4, portion_lines, align="R")
        draw_column(5, transfer_lines, align="L")
        draw_column(6, note_lines, align="L")

        pdf.set_xy(x_start, y_start + row_height)

    # ---------- Present owners section ----------
    pdf.ln(10)
    pdf.set_font("Arial", size=11)
    pdf.cell(
        0,
        8,
        f"Present owners of the property are :",
        ln=1,
    )

    pdf.ln(3)

    owner_headers = ["S.No.", "Name", "Portion"]
    owner_col_widths = [
        15,                  # S.No.
        page_width - 15 - 25,  # Name
        25,                  # Portion
    ]

    pdf.set_font("Arial", "B", 10)
    pdf.set_x(pdf.l_margin + inner_margin)
    for w, header in zip(owner_col_widths, owner_headers):
        pdf.cell(w, 8, header, border=1, align="C")
    pdf.ln(8)

    pdf.set_font("Arial", "", 10)
    for idx, row in enumerate(current_rows, start=1):
        pdf.set_x(pdf.l_margin + inner_margin)
        serial = f"{idx}."
        name = row.get("owner_name") or row.get("name") or ""
        portion = (
            f'{row.get("buyer_portion"):.2f}'
            if isinstance(row.get("buyer_portion"), (int, float))
            else str(row.get("buyer_portion") or "")
        )

        pdf.cell(owner_col_widths[0], 8, serial, border=1)
        pdf.cell(owner_col_widths[1], 8, name, border=1)
        pdf.cell(owner_col_widths[2], 8, portion, border=1, align="R")
        pdf.ln(8)

    # ---------- Share certificate section ----------
    pdf.ln(10)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, "Society membership details:", ln=1)
    pdf.ln(3)

    if share_rows:
        share_headers = [
            "S.No.",
            "Certificate No.",
            "Member Name",
            "Date of Transfer",
        ]
        share_col_widths = [
            10,   # S.No.
            30,   # Cert no
            60,   # Member
            page_width - (10 + 30 + 60),  # Date of Transfer (rest of the width)
        ]

        pdf.set_font("Arial", "B", 10)
        pdf.set_x(pdf.l_margin + inner_margin)
        for w, header in zip(share_col_widths, share_headers):
            pdf.cell(w, 8, header, border=1, align="C")
        pdf.ln(8)

        pdf.set_font("Arial", "", 9)
        for idx, row in enumerate(share_rows, start=1):
            pdf.set_x(pdf.l_margin + inner_margin)
            serial = f"{idx}."
            cert_no = row.get("certificate_number") or ""
            member_name = row.get("member_name") or ""
            d_transfer = (row.get("date_of_transfer") or "")[:10]

            pdf.cell(share_col_widths[0], 8, serial, border=1)
            pdf.cell(share_col_widths[1], 8, cert_no, border=1)
            pdf.cell(share_col_widths[2], 8, member_name, border=1)
            pdf.cell(share_col_widths[3], 8, d_transfer, border=1)
            pdf.ln(8)
    else:
        pdf.set_font("Arial", "I", 9)
        pdf.cell(0, 6, "No share certificate records found for this property.", ln=1)


    # ---------- Club membership section ----------
    pdf.ln(10)
    pdf.set_font("Arial", size=11)
    pdf.cell(0, 8, "Club membership details:", ln=1)
    pdf.ln(3)

    if club_rows:
        club_headers = [
            "S.No.",
            "Membership No.",
            "Member Name",
            "Allocation Date",
        ]
        club_col_widths = [
            10,   # S.No.
            30,   # Membership no
            60,   # Member
            page_width - (10 + 30 + 60),  # Allocation date
        ]

        pdf.set_font("Arial", "B", 10)
        pdf.set_x(pdf.l_margin + inner_margin)
        for w, header in zip(club_col_widths, club_headers):
            pdf.cell(w, 8, header, border=1, align="C")
        pdf.ln(8)

        pdf.set_font("Arial", "", 9)
        for idx, row in enumerate(club_rows, start=1):
            pdf.set_x(pdf.l_margin + inner_margin)
            serial = f"{idx}."
            membership_no = row.get("membership_number") or ""
            member_name = row.get("member_name") or ""
            alloc_date = (row.get("allocation_date") or "")[:10]

            pdf.cell(club_col_widths[0], 8, serial, border=1)
            pdf.cell(club_col_widths[1], 8, membership_no, border=1)
            pdf.cell(club_col_widths[2], 8, member_name, border=1)
            pdf.cell(club_col_widths[3], 8, alloc_date, border=1)
            pdf.ln(8)
    else:
        pdf.set_font("Arial", "I", 9)
        pdf.cell(0, 6, "No club membership records found for this property.", ln=1)

        pdf.cell(0, 6, "No club membership records found for this property.", ln=1)

    # Save PDF
    safe_name = pra.replace("|", "_").replace(" ", "_")
    pdf_filename = f"property_note_{safe_name}.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)
    pdf.output(pdf_path)

    # We don't actually need textual summary anymore; keep empty string for compatibility
    summary_text = ""
    return summary_text, pdf_path, current_rows, history_rows
