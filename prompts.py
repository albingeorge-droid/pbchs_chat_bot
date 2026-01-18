from __future__ import annotations

# =========================
# Query classification
# =========================

QUERY_CLASSIFICATION_SYSTEM_PROMPT = """
You are an intent classifier for a Punjabi Bagh property-ownership chatbot.

Classify ONLY the CURRENT user message into exactly one label:

1) "property_talk"
   The user is asking about Punjabi/Punjabhi Bagh Housing Society property data such as:
   - plots, plot numbers, road numbers, area East/West
   - PRA (e.g. "47|77|Punjabi Bagh West")
   - file numbers, file names
   - current owner, previous owners, buyers, sellers
   - sale deeds, transactions, stamp duty
   - share certificates, club memberships
   - contact details, occupation, address, pan, aadhar, date of birth(dob), phone, email, family members
   - construction details, legal details related to the society records

   IMPORTANT:
   - If the user asks about plots, properties, owners, transactions, files, PRA, or similar
     WITHOUT naming Punjabi Bagh explicitly, STILL treat it as "property_talk".
   - Assume all property-related questions are about Punjabi/Punjabhi Bagh by default,
     unless the user clearly talks about some other city/place.
    - If a person name has already appeared in recent conversation
    in relation to Punjabi Bagh properties, owners, or contact details,
    then follow-up questions about that person (such as date of birth,
    phone, email, PAN, Aadhaar, address, occupation) MUST be classified
    as "property_talk", even if the current message mentions only the name.

   Example:
   - "how many plots are there"  --> label: "property_talk"

2) "small_talk"
   Greetings, thanks, chitchat, questions about you as an AI, or generic talk
   (e.g., "hi", "how are you", "tell me a joke", "thanks").

3) "irrelevant_question"
   Anything outside this domain (weather, cricket, coding help, random facts, etc.)
   IMPORTANT:
   - If the user asks to DELETE / UPDATE / APPEND / REMOVE / DROP / INSERT / EDIT / CHANGE any data,
     classify as "irrelevant_question".

Use recent chat history ONLY to resolve pronouns and follow-ups.
Example: if earlier they discussed a plot and now ask "Who owns it now?",
that is still "property_talk".

You MUST respond with valid JSON ONLY (no markdown, no extra text):
{"label": "...", "reason": "short explanation"}

Rules for the "reason":
- Keep it short.
- Do NOT use the words: "database", "SQL", "system", "query", "JSON".
- Write the reason in normal human language.

Where "label" is exactly one of:
- "property_talk"
- "small_talk"
- "irrelevant_question"
""".strip()



# =========================
# Small-talk & out-of-scope replies
# =========================

SMALL_TALK_SYSTEM_PROMPT = """
You are a helpful assistant for the Punjabi Bagh Housing Society property chatbot.
Act like a normal human.

If the user's query is classified as "small_talk", respond in exactly TWO lines:

Line 1: Give a short, friendly small-talk reply (max 1 sentence).
Line 2: Exactly this text (match spelling/case exactly):
Ask anything related to Punjabi Bagh Housing Society

Rules:
- Do not add any extra lines.
- Do not add bullet points.
- Do not ask follow-up questions.
- Do not mention the words: "database", "SQL", "system", "query", "JSON".
- Keep line 1 concise.
""".strip()


OUT_OF_SCOPE_SYSTEM_PROMPT = """
You are a helpful assistant for the Punjabi Bagh Housing Society property chatbot.
Act like a normal human.

If the user's query is classified as "irrelevant_question", respond with exactly ONE line:

Exactly this text (match spelling/case exactly):
This is an irrelevant question please ask question related to Punjabi Bagh Housing Society

Rules:
- Always output the exact line above, even if the user asks to delete/update/append/remove/drop/insert/edit/change something.
- Do not add any extra text or punctuation.
- Do not add a second line.
- Do not explain why it's irrelevant.
- Do not mention the words: "database", "SQL", "system", "query", "JSON".
""".strip()


# ---- Schema descriptions for Chroma + SQL generation ----
# You can expand / refine these descriptions as needed.


TABLE_SCHEMAS = [
    {
        "table": "properties",
        "description": """
Core property table containing high-level property information such pra_, file_no, file_name.
Columns:
- id (String(36), PK): UUID primary key stored as string
- pra_ (String(255), nullable): Property Reference Address/identifier (e.g. '47|77|Punjabi Bagh West')-> it is the combination of plot_no, road_no and street_name
- file_no (String(255), nullable): Internal file tracking number->basically the initial integer from the file_name
- file_name (String(255), nullable): Name or code of the property file-> its is the pdf name from which a single property data is extracted.
- file_link (Text, nullable): URL/path to property documentation PDF
- qc_status (Enum, required): Quality control status - values: 'raw-json', 'manual-check-1', 'images-mapped', 'manual-check-2', 'client-check'
   Relationships:
   - One Property has one PropertyAddress (address).
   - One Property has many OwnershipRecords.
   - One Property has one CurrentOwner.
   - One Property has one ConstructionDetails.
   - One Property has one LegalDetails.
   - One Property has many ShareCertificates.
   - One Property has many ClubMemberships.
   - One Property has many MiscDocuments.
        """.strip(),
    },
    {
        "table": "property_addresses",
        "description": """
details like plot number, road number, street name , plot size
Columns:
- id (String(36), PK): UUID primary key
- property_id (String(36), FK, required): References properties.id
- plot_no (String(100), nullable): Plot number identifier
- road_no (String(100), nullable): Road number where property is located
- street_name (String(255), nullable): Street name
- initial_plot_size (String(100), nullable): Original size of the plot
- source_page (JSON, list): Page numbers from source documents where this info was found
- flag (Enum): Status flag - 'Pending' or 'Completed'
Relationships:
- Many-to-one: property (back to properties table)
        """.strip(),
    },
    {
        "table": "persons",
        "description": """
Individual persons involved in property transactions (buyers, sellers, members) and their contact details like address , phone number, email,
pan number, aadhaar number,occupation(what kind of work they do)
Columns:
- id (String(36), PK): UUID primary key
- pra (String(255), nullable): Person reference address/identifier
- name (String(255), required): Full name of the person
- dob (String(50), nullable): Date of birth as string
- family_members (JSON, list): List of family member names
- address (Text, nullable): Residential address
- phone_number (String(50), nullable): Contact phone number
- email (String(255), nullable): Email address
- pan (String(50), nullable): PAN card number
- aadhaar (String(50), nullable): Aadhaar card number
- img_link (Text, nullable): URL to person's image/photo
- occupation (String(255), nullable): Professional occupation
- source_page (JSON, list): Source document page references
- person_source (String(255), required): Origin/source of person record
- flag (Enum): Status flag - 'Pending' or 'Completed'
Relationships:
- One-to-many: ownerships_bought (as buyer), sale_deeds, share_certificates, club_memberships
- Many-to-many: sold_in_ownerships (as seller via ownership_sellers), sold_in_current_owner (via current_owner_sellers)
- One Person can buy many OwnershipRecords (ownerships_bought).
- One Person can be linked to many SaleDeeds.
- One Person can have many ShareCertificates (member).
- One Person can have many ClubMemberships (member).
        """.strip(),
    },
    {
        "table": "ownership_records",
        "description": """
Historical ownership records for properties.
buyer portion ,transfer type like sale ,gift ,inheritance etc.
notes about the transaction,
Columns:
- id (String(36), PK): UUID primary key
- property_id (String(36), FK, required): References properties.id
- buyer_id (String(36), FK, nullable): References persons.id for the buyer
- sale_deed_id (String(36), FK, nullable): References sale_deeds.id
- transfer_type (String(255), nullable): Type of ownership transfer (e.g., sale, gift, inheritance)
- buyer_portion (JSON, list, nullable): List describing portion/share acquired by buyer
- total_stamp_duty_paid (String(255), nullable): Total stamp duty amount paid
- notes (Text, nullable): Additional notes about the transaction
- source_page (JSON, list): Source document page references
- flag (Enum): Status flag - 'Pending' or 'Completed'
Relationships:
- Many-to-one: property, buyer (Person), sale_deed
- Many-to-many: sellers (Person via ownership_sellers association table)
- buyer -> persons.id
- sale_deed -> sale_deeds.id
- sellers: many-to-many via ownership_sellers table
        """.strip(),
    },
    {
        "table": "ownership_sellers",
        "description": """
Association table linking ownership records to multiple sellers.
Columns:
- ownership_id (String(36), FK, PK): References ownership_records.id (CASCADE delete)
- person_id (String(36), FK, PK): References persons.id (CASCADE delete)
Purpose: Handles many-to-many relationship between ownership_records and persons (as sellers)
        """.strip(),
    },
    {
        "table": "current_owners",
        "description": """
Current/latest ownership status for each property basically the current owner of each individual property.
Columns:
- id (String(36), PK): UUID primary key
- property_id (String(36), FK, required): References properties.id
- buyer_id (String(36), FK, nullable): References persons.id for current owner
- buyer_portion (String(100), nullable): Portion/share owned by current buyer
- source_page (JSON, list): Source document page references
- flag (Enum): Status flag - 'Pending' or 'Completed'
Relationships:
- Many-to-one: property, buyer (Person)
- Many-to-many: sellers (Person via current_owner_sellers association table)
        """.strip(),
    },
    {
        "table": "current_owner_sellers",
        "description": """
Association table linking current owners to multiple sellers.
Columns:
- current_owner_id (String(36), FK, PK): References current_owners.id (CASCADE delete)
- person_id (String(36), FK, PK): References persons.id (CASCADE delete)
Purpose: Handles many-to-many relationship between current_owners and persons (as sellers)
        """.strip(),
    },
    {
        "table": "sale_deeds",
        "description": """
Sale deed documents with registration details.
it contains the sale deed number, book number, page number, signing date, registry status, owners portion sold, total property portion sold
Columns:
- id (String(36), PK): UUID primary key
- person_id (String(36), FK, nullable): Primary person associated with deed, references persons.id
- property_id (String(36), FK, nullable): References properties.id
- sale_deed_no (JSON, list): List of sale deed numbers
- book_no (JSON, list): List of registration book numbers
- page_no (JSON, list): List of page numbers in registration books
- signing_date (JSON, list): List of signing dates (string format)
- registry_status (String(255), nullable): Status of registry (e.g., registered, pending)
- owners_portion_sold (JSON, list): List describing portions sold by each owner
- total_property_portion_sold (JSON, list): List of total property portions sold
- source_page (JSON, list): Source document page references
- pdf_link (Text, nullable): URL to sale deed PDF document
- flag (Enum): Status flag - 'Pending' or 'Completed'
Relationships:
- Many-to-one: person, property (optional)
- One-to-one: ownership_record (back reference)
Note: Many fields are JSON lists to handle multiple deed entries in one record
        """.strip(),
    },
    {
        "table": "construction_details",
        "description": """
Construction and built-up area details for properties.
Columns:
- id (String(36), PK): UUID primary key
- property_id (String(36), FK, required): References properties.id
- coverage_built_up_area (String(255), default ''): Built-up area coverage
- circle_rate_colony (String(255), default ''): Government circle rate for the colony
- land_price_per_sqm (String(255), default ''): Land price per square meter
- construction_price_per_sqm (String(255), default ''): Construction cost per square meter
- total_covered_area (String(255), default ''): Total covered area of construction
- source_page (JSON, list): Source document page references
- pdf_link (Text, nullable): URL to construction document PDF
- flag (Enum): Status flag - 'Pending' or 'Completed'
Relationships:
- One-to-one: property (back to properties table)
        """.strip(),
    },
    {
        "table": "legal_details",
        "description": """
Legal information and court cases related to properties.
Columns:
- id (String(36), PK): UUID primary key
- property_id (String(36), FK, required): References properties.id
- registrar_office (String(255), default ''): Name of the registrar office
- court_cases (JSON, list): List of court case details/numbers
- source_page (JSON, list): Source document page references
- pdf_link (Text, nullable): URL to legal document PDF
- flag (Enum): Status flag - 'Pending' or 'Completed'
Relationships:
- One-to-one: property (back to properties table)
        """.strip(),
    },
    {
        "table": "share_certificates",
        "description": """
Cooperative society or company share certificates or society membership linked to properties.
Columns:
- id (String(36), PK): UUID primary key
- certificate_number (String(255), nullable): Share certificate number
- property_id (String(36), FK, required): References properties.id
- member_id (String(36), FK, nullable): References persons.id for shareholder
- date_of_transfer (String(100), nullable): Date when shares were transferred
- date_of_ending (String(100), nullable): Date when certificate validity ended
- notes (Text, nullable): Additional notes about the certificate
- source_page (JSON, list): Source document page references
- pdf_link (Text, nullable): URL to certificate PDF
- flag (Enum): Status flag - 'Pending' or 'Completed'
Relationships:
- Many-to-one: property, member (Person)
        """.strip(),
    },
    {
        "table": "club_memberships",
        "description": """
Club memberships associated with properties.
Columns:
- id (String(36), PK): UUID primary key
- member_id (String(36), FK, required): References persons.id
- property_id (String(36), FK, required): References properties.id
- allocation_date (String(100), nullable): Date membership was allocated
- membership_end_date (String(100), nullable): Expiry date of membership
- membership_number (String(255), nullable): Unique membership identifier
- source_page (JSON, list): Source document page references
- pdf_link (Text, nullable): URL to membership document PDF
- flag (Enum): Status flag - 'Pending' or 'Completed'
Relationships:
- Many-to-one: member (Person), property
        """.strip(),
    },
    {
        "table": "misc_documents",
        "description": """
Miscellaneous documents related to properties.
Columns:
- id (String(36), PK): UUID primary key
- property_id (String(36), FK): References properties.id
- pra (String(255), required): Property reference address/identifier
Relationships:
- Many-to-one: property
Purpose: Stores references to additional property documents not covered by other tables
        """.strip(),
    },

]

# ---- Your SQL examples copied verbatim ----

SQL_EXAMPLES = [
    # ========== CURRENT OWNER QUERIES ==========
    {
        "id": "ex1",
        "question": "Who are the current owners of plot 30 road 14",
        "tables": ["properties", "current_owners", "persons", "property_addresses"],
        "sql": """
SELECT T1.file_no, T2.plot_no, T2.road_no, 
       T4.name AS current_owner_name, T3.buyer_portion
FROM properties AS T1
JOIN property_addresses AS T2 ON T1.id = T2.property_id
JOIN current_owners AS T3 ON T1.id = T3.property_id
JOIN persons AS T4 ON T3.buyer_id = T4.id
WHERE T2.plot_no = '30' AND T2.road_no = '14'
LIMIT 50;
        """.strip(),
    },
    {
        "id": "ex2",
        "question": "Who currently owns file number 3447?",
        "tables": ["properties", "current_owners", "persons"],
        "sql": """
SELECT T2.name, T1.buyer_portion
FROM current_owners AS T1
JOIN persons AS T2 ON T1.buyer_id = T2.id
JOIN properties AS T3 ON T1.property_id = T3.id
WHERE T3.file_no = '3447'
LIMIT 50;
        """.strip(),
    },
    {
        "id": "ex3",
        "question": "Show me all properties owned by Davinder Sodhi",
        "tables": ["properties", "current_owners", "persons", "property_addresses"],
        "sql": """
SELECT 
       T2.plot_no, T2.road_no,
       T4.name AS owner_name, T3.buyer_portion
FROM properties AS T1
JOIN property_addresses AS T2 ON T1.id = T2.property_id
JOIN current_owners AS T3 ON T1.id = T3.property_id
JOIN persons AS T4 ON T3.buyer_id = T4.id
WHERE T4.name ILIKE '%Davinder Sodhi%'
LIMIT 50;
        """.strip(),
    },
    {
        "id": "ex4",
        "question": "What is the complete ownership history of plot 30 road 14?",
        "tables": ["properties", "ownership_records", "persons", "sale_deeds", "ownership_sellers","property_addresses"],
        "sql": """
SELECT T1.file_no, T2.plot_no, T2.road_no,
       T3.transfer_type, T3.buyer_portion,
       T4.name AS buyer_name,
       T6.sale_deed_no, T6.signing_date, 
       T7.name AS seller_name
FROM properties AS T1
JOIN property_addresses AS T2 ON T1.id = T2.property_id
LEFT JOIN ownership_records AS T3 ON T1.id = T3.property_id
LEFT JOIN persons AS T4 ON T3.buyer_id = T4.id
LEFT JOIN sale_deeds AS T6 ON T3.sale_deed_id = T6.id
LEFT JOIN ownership_sellers AS T5 ON T3.id = T5.ownership_id
LEFT JOIN persons AS T7 ON T5.person_id = T7.id
WHERE T2.plot_no = '30' AND T2.road_no = '14'
LIMIT 50;
        """.strip(),
    },
    {
        "id": "ex5",
        "question": "give all the transactions for the plot 30 road 14?",
        "tables": ["properties", "ownership_records", "persons", "sale_deeds", "ownership_sellers","property_addresses"],
        "sql": """
SELECT T1.file_no, T2.plot_no, T2.road_no,
       T3.transfer_type, T3.buyer_portion,
       T4.name AS buyer_name,
       T6.sale_deed_no, T6.signing_date, 
       T7.name AS seller_name
FROM properties AS T1
JOIN property_addresses AS T2 ON T1.id = T2.property_id
LEFT JOIN ownership_records AS T3 ON T1.id = T3.property_id
LEFT JOIN persons AS T4 ON T3.buyer_id = T4.id
LEFT JOIN sale_deeds AS T6 ON T3.sale_deed_id = T6.id
LEFT JOIN ownership_sellers AS T5 ON T3.id = T5.ownership_id
LEFT JOIN persons AS T7 ON T5.person_id = T7.id
WHERE T2.plot_no = '30' AND T2.road_no = '14'
LIMIT 50;
        """.strip(),
    },
    {
        "id": "ex6",
        "question": "List all transactions involving Davinder Sodh",
        "tables": ["properties", "ownership_records", "persons", "sale_deeds", "ownership_sellers"],
        "sql": """
SELECT T1.file_no, T3.name AS buyer_name, T5.name AS seller_name, 
       (T4.signing_date->>0) AS signing_date, T1.file_name, T2.buyer_portion, T2.notes
FROM properties AS T1
JOIN ownership_records AS T2 ON T1.id = T2.property_id
JOIN persons AS T3 ON T2.buyer_id = T3.id
JOIN sale_deeds AS T4 ON T2.sale_deed_id = T4.id
JOIN ownership_sellers AS T6 ON T6.ownership_id = T2.id
JOIN persons AS T5 ON T5.id = T6.person_id
WHERE T3.name ILIKE '%Davinder Sodh%' OR T5.name ILIKE '%Davinder Sodh%'
LIMIT 50;
        """.strip(),
    },
    {
        "id": "ex7",
        "question": "How many transactions happened before 2005?",
        "tables": ["ownership_records", "sale_deeds"],
        "sql": """
SELECT COUNT(*) AS total_count
FROM ownership_records AS T1
JOIN sale_deeds AS T2 ON T1.sale_deed_id = T2.id
WHERE (T2.signing_date->>0) IS NOT NULL
  AND (T2.signing_date->>0) != ''
  AND to_date((T2.signing_date->>0), 'DD/MM/YYYY') < '2005-01-01';
        """.strip(),
    },
    {
        "id": "ex8",
        "question": "How many properties are there",
        "tables": ["properties"],
        "sql": """
SELECT COUNT(*) AS total_count
FROM properties;
        """.strip(),
    },
    {
        "id": "ex9",
        "question": "How many transactions involved Davinder Sodh?",
        "tables": ["ownership_records", "persons", "ownership_sellers"],
        "sql": """
SELECT COUNT(*) AS total_count
FROM ownership_records AS T1
JOIN persons AS T2 ON T1.buyer_id = T2.id
LEFT JOIN ownership_sellers AS T3 ON T3.ownership_id = T1.id
LEFT JOIN persons AS T4 ON T4.id = T3.person_id
WHERE T2.name ILIKE '%Davinder Sodh%' OR T4.name ILIKE '%Davinder Sodh%';
        """.strip(),
    },
    {
        "id": "ex10",
        "question": "What is the occupation of Davinder Sodhi?",
        "tables": ["persons"],
        "sql": """
SELECT name, occupation, phone_number, email
FROM persons
WHERE name ILIKE '%Davinder Sodhi%'
LIMIT 50;
        """.strip(),
    },
    {
        "id": "ex11",
        "question": "Show me contact details for Davinder Sodhi",
        "tables": ["persons"],
        "sql": """
SELECT name, phone_number, email, address, dob,pan, aadhaar, occupation
FROM persons
WHERE name ILIKE '%Davinder Sodhi%'
LIMIT 50;
        """.strip(),
    },
    {
        "id": "ex12",
        "question": "Find the person with PAN number AARPS7445L",
        "tables": ["persons"],
        "sql": """
SELECT name, pan, phone_number, address
FROM persons
WHERE pan ILIKE '%AARPS7445L%'
LIMIT 50;
        """.strip(),
    },
    {
        "id": "ex13",
        "question": "Who are people whose occupation is Business?",
        "tables": ["persons"],
        "sql": """
SELECT name, occupation, phone_number
FROM persons
WHERE occupation ILIKE '%Business%'
LIMIT 50;
        """.strip(),
    },

    {
        "id": "ex14",
        "question": "What is the plot size for plot 30 road 14",
        "tables": ["properties", "property_addresses"],
        "sql": """
SELECT T1.file_no, T2.plot_no, T2.road_no, T2.initial_plot_size
FROM properties AS T1
JOIN property_addresses AS T2 ON T1.id = T2.property_id
WHERE T2.plot_no = '30' AND T2.road_no = '14'
LIMIT 50;
        """.strip(),
    },
    {
        "id": "ex15",
        "question": "Show me all properties near road 14",
        "tables": ["properties", "property_addresses"],
        "sql": """
SELECT T1.file_no, T2.plot_no, T2.road_no, T2.street_name ,T2.initial_plot_size
FROM properties AS T1
JOIN property_addresses AS T2 ON T1.id = T2.property_id
WHERE T2.road_no = '14'
LIMIT 50;
        """.strip(),
    },
    {
        "id": "ex16",
        "question": "What properties are in Punjabi Bagh West?",
        "tables": ["properties"],
        "sql": """
SELECT file_no, pra_
FROM properties
WHERE pra_ ILIKE '%Punjabi Bagh West%'
LIMIT 50;
        """.strip(),
    },

    {
        "id": "ex17",
        "question": "Show construction details for plot 30 road 14",
        "tables": ["properties", "construction_details", "construction_details", "property_addresses"],
        "sql": """
SELECT T1.file_no, T2.plot_no, T2.road_no, T2.street_name, T2.initial_plot_size,
       T3.coverage_built_up_area, T3.circle_rate_colony, T3.land_price_per_sqm, 
       T3.construction_price_per_sqm, T3.total_covered_area
FROM properties AS T1
JOIN property_addresses AS T2 ON T1.id = T2.property_id
JOIN construction_details AS T3 ON T1.id = T3.property_id
WHERE T2.plot_no = '30' AND T2.road_no = '14'
LIMIT 50;
        """.strip(),
    },
    {
        "id": "ex18",
        "question": "What is the built-up area for plot 30 road 14",
        "tables": ["properties", "construction_details","construction_details", "property_addresses"],
        "sql": """
SELECT T1.file_no, T2.plot_no, T2.road_no, T2.street_name,
       T3.coverage_built_up_area, T3.total_covered_area
FROM properties AS T1
JOIN property_addresses AS T2 ON T1.id = T2.property_id
JOIN construction_details AS T3 ON T1.id = T3.property_id
WHERE T2.plot_no = '30' AND T2.road_no = '14'
LIMIT 50;
        """.strip(),
    },
    {
        "id": "ex19",
        "question": "Show me properties with land price greater than 50000 per sqm",
        "tables": ["properties", "construction_details"],
        "sql": """
SELECT T1.file_no, T1.pra_, T2.land_price_per_sqm, T2.construction_price_per_sqm
FROM properties AS T1
JOIN construction_details AS T2 ON T1.id = T2.property_id
WHERE T2.land_price_per_sqm != '' 
  AND CAST(T2.land_price_per_sqm AS NUMERIC) > 50000
LIMIT 50;
        """.strip(),
    },

    {
        "id": "ex20",
        "question": "What are the legal details for plot 30 road 14",
        "tables": ["properties", "legal_details"],
        "sql": """
SELECT T1.file_no, T2.plot_no, T2.road_no, T2.street_name,
       T3.registrar_office, T3.court_cases
FROM properties AS T1
JOIN property_addresses AS T2 ON T1.id = T2.property_id
JOIN legal_details AS T3 ON T1.id = T3.property_id
WHERE T2.plot_no = '30' AND T2.road_no = '14'
LIMIT 50;
        """.strip(),
    },
    {
        "id": "ex21",
        "question": "Show all properties with court cases",
        "tables": ["properties", "legal_details"],
        "sql": """
SELECT T1.file_no, T1.pra_, T2.court_cases, T2.registrar_office
FROM properties AS T1
JOIN legal_details AS T2 ON T1.id = T2.property_id
WHERE T2.court_cases IS NOT NULL 
  AND T2.court_cases::text != '[]'
LIMIT 50;        """.strip(),
    },
    {
        "id": "ex22",
        "question": "Which registrar office handles plot 30 road 14",
        "tables": ["properties", "legal_details", "property_addresses"],
        "sql": """
SELECT T1.file_no, T2.plot_no, T2.road_no, T2.street_name,
       T3.registrar_office
FROM properties AS T1
JOIN property_addresses AS T2 ON T1.id = T2.property_id
JOIN legal_details AS T3 ON T1.id = T3.property_id
WHERE T2.plot_no = '30' AND T2.road_no = '14'
LIMIT 50;
        """.strip(),
    },

    {
        "id": "ex23",
        "question": "List all transactions after January 2010",
        "tables": ["properties", "ownership_records", "sale_deeds", "persons", "ownership_sellers"],
        "sql": """
SELECT T1.file_no, T1.pra_, T3.name AS buyer_name, T5.name AS seller_name, 
       (T4.signing_date->>0) AS signing_date,T2.buyer_portion
FROM properties AS T1
JOIN ownership_records AS T2 ON T1.id = T2.property_id
JOIN persons AS T3 ON T2.buyer_id = T3.id
JOIN sale_deeds AS T4 ON T2.sale_deed_id = T4.id
JOIN ownership_sellers AS T6 ON T6.ownership_id = T2.id
JOIN persons AS T5 ON T5.id = T6.person_id
WHERE (T4.signing_date->>0) IS NOT NULL
  AND (T4.signing_date->>0) != ''
  AND to_date((T4.signing_date->>0), 'DD/MM/YYYY') > '2010-01-01'
LIMIT 50;
        """.strip(),
    },
    {
        "id": "ex24",
        "question": "Show transactions between 2015 and 2020",
        "tables": ["properties", "ownership_records", "sale_deeds", "persons", "ownership_sellers"],
        "sql": """
SELECT T1.file_no,T1.pra_,T3.name AS buyer_name, T5.name AS seller_name, T2.buyer_portion,
       (T4.signing_date->>0) AS signing_date
FROM properties AS T1
JOIN ownership_records AS T2 ON T1.id = T2.property_id
JOIN persons AS T3 ON T2.buyer_id = T3.id
JOIN sale_deeds AS T4 ON T2.sale_deed_id = T4.id
JOIN ownership_sellers AS T6 ON T6.ownership_id = T2.id
JOIN persons AS T5 ON T5.id = T6.person_id
WHERE (T4.signing_date->>0) IS NOT NULL
  AND (T4.signing_date->>0) != ''
  AND to_date((T4.signing_date->>0), 'DD/MM/YYYY') BETWEEN '2015-01-01' AND '2020-12-31'
LIMIT 50;        """.strip(),
    },
    {
        "id": "ex25",
        "question": "What transactions happened in 2018?",
        "tables": ["properties", "ownership_records", "sale_deeds", "persons", "ownership_sellers"],
        "sql": """
SELECT T1.file_no,T1.pra_,T3.name AS buyer_name, T5.name AS seller_name, T2.buyer_portion,
       (T4.signing_date->>0) AS signing_date
FROM properties AS T1
JOIN ownership_records AS T2 ON T1.id = T2.property_id
JOIN persons AS T3 ON T2.buyer_id = T3.id
JOIN sale_deeds AS T4 ON T2.sale_deed_id = T4.id
JOIN ownership_sellers AS T6 ON T6.ownership_id = T2.id
JOIN persons AS T5 ON T5.id = T6.person_id
WHERE (T4.signing_date->>0) IS NOT NULL
  AND (T4.signing_date->>0) != ''
  AND EXTRACT(YEAR FROM to_date((T4.signing_date->>0), 'DD/MM/YYYY')) = 2018
LIMIT 50;
        """.strip(),
    },

    {
        "id": "ex26",
        "question": "Show society membership details for member Davinder Sodhi",
        "tables": ["share_certificates", "persons", "properties"],
        "sql": """
SELECT T3.file_no, T1.certificate_number, T2.name,  
       T1.date_of_transfer, T1.date_of_ending
FROM share_certificates AS T1
JOIN persons AS T2 ON T1.member_id = T2.id
JOIN properties AS T3 ON T1.property_id = T3.id
WHERE T2.name ILIKE '%Rajesh Kumar%'
LIMIT 50;
        """.strip(),
    },
    {
        "id": "ex27",
        "question": "Find society members for the plot 30 road 14",
        "tables": ["share_certificates", "persons", "properties", "property_addresses"],
        "sql": """
SELECT T1.file_no, T2.plot_no, T2.road_no, T2.street_name,
       T3.certificate_number, T3.date_of_transfer, T3.date_of_ending, T3.notes,
       T4.name AS member_name
FROM properties AS T1
JOIN property_addresses AS T2 ON T1.id = T2.property_id
JOIN share_certificates AS T3 ON T1.id = T3.property_id
JOIN persons AS T4 ON T3.member_id = T4.id
WHERE T2.plot_no = '30' AND T2.road_no = '14'
LIMIT 50;        """.strip(),
    },

    {
        "id": "ex28",
        "question": "Show club membership details for the plot 30 road 14",
        "tables": ["club_memberships", "persons", "properties"],
        "sql": """
SELECT T1.file_no, T2.plot_no, T2.road_no, T2.street_name,
       T3.membership_number, T3.allocation_date, T3.membership_end_date,
       T4.name AS member_name
FROM properties AS T1
JOIN property_addresses AS T2 ON T1.id = T2.property_id
JOIN club_memberships AS T3 ON T1.id = T3.property_id
JOIN persons AS T4 ON T3.member_id = T4.id
WHERE T2.plot_no = '30' AND T2.road_no = '14'
LIMIT 50;
        """.strip(),
    },
    {
        "id": "ex29",
        "question": "What is the file number of plot plot number 10 road 23?",
        "tables": ["properties","property_addresses"],
        "sql": """
SELECT T1.file_no
FROM properties AS T1
JOIN property_addresses AS T2 ON T1.id = T2.property_id
WHERE T2.plot_no = '10' AND T2.road_no = '23'
LIMIT 1;
        """.strip(),
    },

    {
        "id": "ex30",
        "question": "List all properties with their current owners and plot sizes",
        "tables": ["properties", "current_owners", "persons", "property_addresses"],
        "sql": """
SELECT T1.file_no, T1.pra_, T3.name AS owner_name, T2.buyer_portion ,T4.initial_plot_size
FROM properties AS T1
LEFT JOIN current_owners AS T2 ON T1.id = T2.property_id
LEFT JOIN persons AS T3 ON T2.buyer_id = T3.id
LEFT JOIN property_addresses AS T4 ON T1.id = T4.property_id
LIMIT 50;        """.strip(),
    },

    {
        "id": "ex31",
        "question": "How many properties does each person own?",
        "tables": ["current_owners", "persons"],
        "sql": """
SELECT T2.name, COUNT(*) AS property_count
FROM current_owners AS T1
JOIN persons AS T2 ON T1.buyer_id = T2.id
GROUP BY T2.name
ORDER BY property_count DESC
LIMIT 50;
        """.strip(),
    },
    {
        "id": "ex32",
        "question": "Count transactions by year",
        "tables": ["ownership_records", "sale_deeds"],
        "sql": """
SELECT EXTRACT(YEAR FROM to_date((T2.signing_date->>0), 'DD/MM/YYYY')) AS year, 
       COUNT(*) AS transaction_count
FROM ownership_records AS T1
JOIN sale_deeds AS T2 ON T1.sale_deed_id = T2.id
WHERE (T2.signing_date->>0) IS NOT NULL
  AND (T2.signing_date->>0) != ''
GROUP BY year
ORDER BY year DESC
LIMIT 50;
        """.strip(),
    },
    {
        "id": "ex33",
        "question": "how many properties come in Punjabi Bagh East?",
        "tables": ["properties"],
        "sql": """
SELECT COUNT(*) AS property_count
FROM properties
WHERE pra_ LIKE '%Punjabi Bagh East%';
        """.strip(),
    },
    {
        "id": "ex34",
        "question": "How many properties does Yogesh Berry own and also name that property",
        "tables": ["properties","property_addresses","current_owners","persons"],
        "sql": """
SELECT COUNT(T1.id) AS total_properties,
       T1.file_no, T2.plot_no, T2.road_no, T2.street_name,
       T4.name AS owner_name, T3.buyer_portion
FROM properties AS T1
JOIN property_addresses AS T2 ON T1.id = T2.property_id
JOIN current_owners AS T3 ON T1.id = T3.property_id
JOIN persons AS T4 ON T3.buyer_id = T4.id
WHERE T4.name ILIKE '%Yogesh Berry%'
GROUP BY T2.plot_no, T2.road_no, T2.street_name, T4.name, T3.buyer_portion
LIMIT 50;
        """.strip(),
    },
{
    "id": "ex35",
    "question": "How many properties are there and how many are in Punjabi Bagh East?",
    "tables": ["properties", "property_addresses"],
    "sql": """
SELECT 
    COUNT(DISTINCT T1.id) AS total_properties,
    COUNT(DISTINCT CASE WHEN T2.street_name ILIKE '%Punjabi Bagh East%' THEN T1.id END) AS properties_in_punjabi_bagh_east
FROM properties AS T1
LEFT JOIN property_addresses AS T2 ON T1.id = T2.property_id
LIMIT 50;
    """.strip(),
},
{
    "id": "ex36",
    "question": "How many transactions were done before year 2000? Also tell what were they",
    "tables": ["ownership_records", "sale_deeds", "persons", "ownership_sellers", "properties", "property_addresses"],
    "sql": """
SELECT T6.file_no, T2.signing_date, T1.transfer_type, 
       T3.name AS buyer_name, 
       T5.name AS seller_name,
       T1.buyer_portion,
       T7.plot_no, T7.road_no, T7.street_name
FROM ownership_records AS T1
LEFT JOIN sale_deeds AS T2 ON T1.sale_deed_id = T2.id
LEFT JOIN persons AS T3 ON T1.buyer_id = T3.id
LEFT JOIN ownership_sellers AS T4 ON T1.id = T4.ownership_id
LEFT JOIN persons AS T5 ON T4.person_id = T5.id
LEFT JOIN properties AS T6 ON T1.property_id = T6.id
LEFT JOIN property_addresses AS T7 ON T6.id = T7.property_id
WHERE to_date(T2.signing_date->>0, 'DD/MM/YYYY') < '2000-01-01'
ORDER BY to_date(T2.signing_date->>0, 'DD/MM/YYYY')
LIMIT 50;
    """.strip(),
},
{
    "id": "ex37",
    "question": "Give me those plot number, road number and buyer and seller name where the current owner is more than one",
    "tables": ["properties", "property_addresses", "current_owners", "persons", "current_owner_sellers"],
    "sql": """
SELECT T1.file_no, T2.plot_no, T2.road_no, T2.street_name,
       T4.name AS current_owner_name,
       T6.name AS seller_name
FROM properties AS T1
JOIN property_addresses AS T2 ON T1.id = T2.property_id
JOIN current_owners AS T3 ON T1.id = T3.property_id
JOIN persons AS T4 ON T3.buyer_id = T4.id
LEFT JOIN current_owner_sellers AS T5 ON T3.id = T5.current_owner_id
LEFT JOIN persons AS T6 ON T5.person_id = T6.id
WHERE T1.id IN (
    SELECT property_id
    FROM current_owners
    GROUP BY property_id
    HAVING COUNT(id) > 1
)
ORDER BY T2.plot_no, T2.road_no
LIMIT 50;
    """.strip(),
},
{
    "id": "ex38",
    "question": "What is the plot where the number of transactions is maximum?",
    "tables": ["properties", "property_addresses", "ownership_records"],
    "sql": """
SELECT T1.file_no, T1.id, T2.plot_no, T2.road_no, COUNT(T3.id) AS transaction_count 
FROM properties AS T1 
JOIN property_addresses AS T2 ON T1.id = T2.property_id 
JOIN ownership_records AS T3 ON T1.id = T3.property_id 
GROUP BY T1.id, T2.plot_no, T2.road_no ORDER BY COUNT(T3.id) DESC LIMIT 1;
    """.strip(),
},
{
    "id": "ex39",
    "question": "What are the top 10 plots according to their size?",
    "tables": ["properties", "property_addresses"],
    "sql": """
SELECT
       T1.file_no,
       T2.plot_no,
       T2.road_no,
       T2.initial_plot_size
FROM properties AS T1
JOIN property_addresses AS T2
  ON T1.id = T2.property_id
WHERE NULLIF(TRIM(T2.initial_plot_size), '') IS NOT NULL
ORDER BY NULLIF(TRIM(T2.initial_plot_size), '')::DECIMAL DESC
LIMIT 10;
    """.strip(),
},
{
    "id": "ex40",
    "question": "What are the plots where court cases are maximum?",
    "tables": ["properties", "legal_details"],
    "sql": """
SELECT T1.file_no, T1.pra_, COUNT(T2.court_cases) AS court_case_count 
FROM properties AS T1 
JOIN legal_details AS T2 ON T1.id = T2.property_id 
WHERE NOT T2.court_cases IS NULL AND CAST(T2.court_cases AS TEXT) <> '[]' 
GROUP BY T1.pra_ ORDER BY COUNT(T2.court_cases) DESC LIMIT 50;
    """.strip(),
},
{
    "id": "ex41",
    "question": "Who were the previous owners of plot 30 on road 14?",
    "tables": ["properties", "property_addresses", "ownership_records","persons","sale_deeds","ownership_sellers"],
    "sql": """
SELECT T1.file_no, T2.plot_no, T2.road_no, T3.transfer_type, T3.buyer_portion, T4.name AS buyer_name, T6.sale_deed_no, T6.signing_date, T7.name AS seller_name 
FROM properties AS T1 
JOIN property_addresses AS T2 ON T1.id = T2.property_id 
LEFT JOIN ownership_records AS T3 ON T1.id = T3.property_id 
LEFT JOIN persons AS T4 ON T3.buyer_id = T4.id 
LEFT JOIN sale_deeds AS T6 ON T3.sale_deed_id = T6.id 
LEFT JOIN ownership_sellers AS T5 ON T3.id = T5.ownership_id 
LEFT JOIN persons AS T7 ON T5.person_id = T7.id 
WHERE T2.plot_no = '30' AND T2.road_no = '14' LIMIT 50;
    """.strip(),
},
{
    "id": "ex42",
    "question": "What transactions were done before the year 2000?",
    "tables": ["properties", "ownership_records","persons","sale_deeds","ownership_sellers"],
    "sql": """
SELECT T1.file_no, T1.pra_, T3.name AS buyer_name, T5.name AS seller_name, T2.buyer_portion, (T4.signing_date ->> 0) AS signing_date 
FROM properties AS T1 
JOIN ownership_records AS T2 ON T1.id = T2.property_id 
JOIN persons AS T3 ON T2.buyer_id = T3.id 
JOIN sale_deeds AS T4 ON T2.sale_deed_id = T4.id 
JOIN ownership_sellers AS T6 ON T6.ownership_id = T2.id 
JOIN persons AS T5 ON T5.id = T6.person_id 
WHERE NOT (T4.signing_date ->> 0) IS NULL AND (T4.signing_date ->> 0) <> '' AND TO_DATE((T4.signing_date ->> 0), 'DD/MM/YYYY') < '2000-01-01' LIMIT 100;
    """.strip(),
},
{
    "id": "ex43",
    "question": "Who is the original owner of plot 30 on road 14?",
    "tables": ["properties", "property_addresses", "ownership_records","persons","sale_deeds","ownership_sellers"],
    "sql": """
SELECT T1.file_no, T2.plot_no, T2.road_no, T3.transfer_type, T3.buyer_portion, T4.name AS buyer_name, T6.sale_deed_no, T6.signing_date, T7.name AS seller_name 
FROM properties AS T1 
JOIN property_addresses AS T2 ON T1.id = T2.property_id 
LEFT JOIN ownership_records AS T3 ON T1.id = T3.property_id 
LEFT JOIN persons AS T4 ON T3.buyer_id = T4.id LEFT JOIN sale_deeds AS T6 ON T3.sale_deed_id = T6.id 
LEFT JOIN ownership_sellers AS T5 ON T3.id = T5.ownership_id 
LEFT JOIN persons AS T7 ON T5.person_id = T7.id 
WHERE T2.plot_no = '30' AND T2.road_no = '14' LIMIT 50;
    """.strip(),
},
{
    "id": "ex44",
    "question": "What is the transaction history of plot 5 on road East Avenu where the transfer type is sale?",
    "tables": ["properties","ownership_records","persons","sale_deeds","ownership_sellers"],
    "sql": """
SELECT T1.file_no, T2.plot_no, T2.road_no, T3.transfer_type, T3.buyer_portion, T4.name AS buyer_name, T6.sale_deed_no, T6.signing_date, T7.name AS seller_name 
FROM properties AS T1 JOIN property_addresses AS T2 ON T1.id = T2.property_id 
LEFT JOIN ownership_records AS T3 ON T1.id = T3.property_id AND T3.transfer_type ILIKE '%sale%' 
LEFT JOIN persons AS T4 ON T3.buyer_id = T4.id 
LEFT JOIN sale_deeds AS T6 ON T3.sale_deed_id = T6.id 
LEFT JOIN ownership_sellers AS T5 ON T3.id = T5.ownership_id LEFT JOIN persons AS T7 ON T5.person_id = T7.id WHERE T2.plot_no = '5' AND T2.road_no = 'East Avenue Road' LIMIT 50;
    """.strip(),
},
{
    "id": "ex45",
    "question": "Give me all the properties where owner's last name is Kohli",
    "tables": ["properties", "property_addresses", "current_owners", "persons"],
    "sql": """
SELECT 
       T1.file_no, T2.plot_no, T2.road_no, T2.street_name, T2.initial_plot_size,
       T4.name AS owner_name, T3.buyer_portion
FROM properties AS T1
JOIN property_addresses AS T2 ON T1.id = T2.property_id
JOIN current_owners AS T3 ON T1.id = T3.property_id
JOIN persons AS T4 ON T3.buyer_id = T4.id
WHERE T4.name ILIKE '%Kohli%'
ORDER BY T2.plot_no, T2.road_no
LIMIT 50;
    """.strip(),
},
{
    "id": "ex46",
    "question": "Give me all the owners who have more than one plot",
    "tables": ["current_owners", "persons", "properties", "property_addresses"],
    "sql": """
SELECT T3.file_no, T2.name AS owner_name, 
       COUNT(DISTINCT T1.property_id) AS total_properties,
       STRING_AGG(DISTINCT T4.plot_no || '|' || T4.road_no || '|' || T4.street_name, ', ') AS properties_owned
FROM current_owners AS T1
JOIN persons AS T2 ON T1.buyer_id = T2.id
JOIN properties AS T3 ON T1.property_id = T3.id
JOIN property_addresses AS T4 ON T3.id = T4.property_id
GROUP BY T2.id, T2.name, T2.phone_number, T2.email, T2.address
HAVING COUNT(DISTINCT T1.property_id) > 1
ORDER BY total_properties DESC
LIMIT 50;
    """.strip(),
},
{
    "id": "ex47",
    "question": "Show the contact details of albin of plot 30 road 14",
    "tables": ["persons", "current_owners", "properties", "property_addresses"],
    "sql": """
SELECT 
     T1.name,
       T1.phone_number,
       T1.email,
       T1.address,
       T1.pan,
       T1.aadhaar,
       T1.occupation,
       T3.plot_no,
       T3.road_no,
       T3.street_name
FROM persons AS T1
JOIN current_owners AS T2 ON T1.id = T2.buyer_id
JOIN property_addresses AS T3 ON T2.property_id = T3.property_id
WHERE T3.plot_no = '30' 
  AND T3.road_no = '14'
  AND LOWER(T1.name) LIKE '%albin%';
    """.strip(),
},

    {
        "id": "ex48",
        "question": "What plots are located near East Avenue Road?",
        "tables": ["properties", "property_addresses"],
        "sql": """
SELECT T1.file_no, T2.plot_no, T2.road_no, T2.street_name ,T2.initial_plot_size
FROM properties AS T1
JOIN property_addresses AS T2 ON T1.id = T2.property_id
WHERE T2.road_no = 'East Avenue Road'
LIMIT 50;
        """.strip(),
    },

    {
        "id": "ex49",
        "question": "What plots are near North Avenue Road?",
        "tables": ["properties", "property_addresses"],
        "sql": """
SELECT T1.file_no, T2.plot_no, T2.road_no, T2.street_name ,T2.initial_plot_size
FROM properties AS T1
JOIN property_addresses AS T2 ON T1.id = T2.property_id
WHERE T2.road_no = 'North Avenue Road'
LIMIT 50;
        """.strip(),
    },




]

# ---- Prompt templates ----

STANDALONE_QUESTION_PROMPT = """
You rewrite a user's message into:
1) a cleaned (normalized) version of the same message, and
2) a standalone English question that can be understood without context.

You are helping with Punjabi Bagh Housing Society property records.

You are given:
- Recent chat history (JSON list of role + content):
{history_json}

- Current user query (the ONLY message you must rewrite):
{user_query}

- Extracted entities from NER (may be partial):
{ner_json}

GOAL
Return ONLY a JSON object with:
- language
- normalized_query (cleaned wording, same meaning)
- standalone_question (English, self-contained)

IMPORTANT RULES (STRICT)
1) Use chat history ONLY when the current user query is vague and missing identifiers.
   Vague means it relies on references like: "this", "that", "it", "this property", "that plot",
   "the above", "same one", etc.

2) If the current user query already contains ANY explicit property identifier, DO NOT use history
   to add more or change to a different property.
   Explicit property identifiers include:
   - PRA like "28|6|Punjabi Bagh East/West"
   - plot number and/or road number (example: "plot 30", "road 14", "plot 30 on road 14")
   - file_no or file_name
   - area: "Punjabi Bagh East" or "Punjabi Bagh West"
   - identifiers inside parentheses such as "(for property PRA 30|14|Punjabi Bagh East)"

3) NEVER add a person name from history.
   Only include person_name in normalized_query/standalone_question if:
   - the CURRENT user query explicitly mentions a person name, OR
   - the CURRENT user query uses person-pronouns: "him", "her", "them", "his", "their",
     "she", "he", "they".
   Otherwise, do not add any "(for person ...)" or any person reference.

4) If identifiers are missing and the query is vague:
   - Use history + NER to fill ONLY the missing property identifier(s) (PRA or plot/road or file).
   - Prefer PRA if available. Otherwise use plot/road. Otherwise file_no/file_name.

5) Do NOT add extra explanations, notes, parentheses, or meta text.
   Output must be ONLY valid JSON (no markdown, no extra keys).

6) **CRITICAL RULE – DO NOT ADD EXTRA HOUSING-SOCIETY CONTEXT**:
   - If the user query does NOT mention "Punjabi Bagh Housing Society", "Housing Society",
     "the society", or "PBHS", do NOT introduce these phrases.
   - However, if the user query DOES include words like "society member", "society members",
     "society membership", "share certificate", or "society shares", you MUST preserve those
     words exactly in BOTH normalized_query and standalone_question.
   - Keep the normalized_query and standalone_question as MINIMAL as possible otherwise.


7) If the user query text (as shown above) contains any explicit property identifier, you MUST
   preserve those identifiers in BOTH normalized_query and standalone_question.
   - You may reorder words slightly for clarity, but you MUST keep the same PRA / plot / road /
     area / file identifiers.
   - You MUST NOT replace explicit identifiers with vague phrases like "this plot",
     "this property", or "that file" in standalone_question.

8) When you use history + NER to resolve a vague question like "this plot" or "this property"
   to a specific property, standalone_question MUST mention that property explicitly
   (e.g. "plot 30 on road 14 in Punjabi Bagh East" or "property 30|14|Punjabi Bagh East"),
   NOT with pronouns.

9) If the user query contains any extra filters or qualifiers, you MUST keep them
   in BOTH normalized_query and standalone_question. Do NOT broaden the question.
   Examples of filters/qualifiers that MUST be preserved include phrases like:
   - "through sale", "via sale", "via inheritance", "through relinquishment",
   - "between 2000 and 2020", "after 2015", "before 2000",
   - "only current owners", "original owner", "first buyer".
   Never drop words like "sale" or a date range when rewriting.

OUTPUT JSON FORMAT (exact keys):
{{
  "language": "detected_language",
  "normalized_query": "cleaned_up_user_query",
  "standalone_question": "self contained English question for the property SQL agent"
}}
""".strip()

SQL_GENERATION_SYSTEM_PROMPT = """
You are an expert PostgreSQL query generator for a Property Ownership system.

Rules:
- Output ONLY a single PostgreSQL SELECT query, ending with a semicolon.
- Never modify data: no INSERT/UPDATE/DELETE/ALTER/DROP/TRUNCATE/GRANT/REVOKE.
- Use the given schema and examples carefully.
- Prefer ILIKE for case-insensitive text search FOR names ONLY.
- If a PRA is given, filter on properties.pra_ .
- File number handling:
  - If a specific file_name or file_no VALUE is given in the question, filter on properties.file_name or properties.file_no
    (use ILIKE only if the value looks partial/contains wildcards).
  - If the question says "with file number" / "include file number" / "show file number" (but does NOT give a specific value),
    then INCLUDE properties.file_no (and optionally properties.file_name) in the SELECT list, and DO NOT add a WHERE filter like file_no IS NOT NULL.

- If a person_name is given, filter on persons.name using ILIKE with wildcards.

- If the question asks for the *current owner* only (and does NOT ask about dates or when the property was purchased / transferred):
  - Use the current_owners table and join to properties and persons.
  - Do NOT join to sale_deeds unless explicitly needed.

- If the question asks for "ownership history" or "transactions":
  - Use ownership_records + sale_deeds + ownership_sellers + persons + properties.
  - Use sale_deeds.signing_date as the transaction date
    (for example, (sale_deeds.signing_date->>0) if it is stored as a JSON list).

- If the question asks for the "most recent owner" AND also asks for a date / dates
  (for example: "most recent owner and date", "latest owner and transaction date",
   "who owns it now and when was it last transferred"):
  - Treat this as an ownership history / transaction query, NOT as a simple current_owners query.
  - Use ownership_records + sale_deeds + persons + properties.
  - Use sale_deeds.signing_date (e.g. (sale_deeds.signing_date->>0)) as the transaction date.
  - For each property, pick the ownership_records row whose signing_date is the latest
    (i.e. the most recent transaction) and return that buyer as the "most recent owner".

- If the question talks about a *club member* or *club membership* (e.g. "club member", "club membership number", "PBCHS club card"):
  - Use the club_memberships table joined with persons and properties.
  - membership_number is stored in club_memberships.membership_number.
  - Use property_addresses and/or properties to locate the correct property if plot/road/PRA are mentioned.

- If the question talks about a *society member* or *society membership* (e.g. "society member", "society membership", "share certificate", "society shares"):
  - Use the share_certificates table joined with persons and properties.
  - The society membership / share certificate number is stored in share_certificates.certificate_number.
  - Use property_addresses and/or properties to locate the correct property if plot/road/PRA are mentioned.

- Never use persons.dob as the date of an ownership change or transaction.
  - persons.dob is ONLY for the person's date of birth, not the purchase date or transfer date.

- When you need to pick the latest / most recent transaction per property:
  - Use a MAX() on (sale_deeds.signing_date->>0) or a suitable subquery / CTE
    to select the row with the largest signing_date for each property.

Important column-specific rule:

- sale_deeds.signing_date is stored as JSON/text like '28/01/1962' (DD/MM/YYYY).
    - Whenever you need a DATE from it, ALWAYS use:
        to_date(alias.signing_date->>0, 'DD/MM/YYYY')
    (where "alias" is the table alias, e.g. sd or sd2).
    - NEVER use CAST(... AS DATE) or ::date on signing_date->>0.

- Column initial_plot_size is TEXT. Whenever you need to order or filter
  numerically on it, first exclude empty strings and cast like this:
  WHERE NULLIF(TRIM(property_addresses.initial_plot_size), '') IS NOT NULL
  ORDER BY NULLIF(TRIM(property_addresses.initial_plot_size), '')::DECIMAL ...



Additional JSON rules:

- ownership_records.buyer_portion is stored as JSON (e.g. ['37.50']).
- NEVER GROUP BY the raw JSON column ownership_records.buyer_portion.
- If you need the value, use (ownership_records.buyer_portion->>0) or
  CAST(ownership_records.buyer_portion->>0 AS numeric) in SELECT or WHERE,
  but avoid grouping by it unless you group by that TEXT/NUMERIC expression.
- In general, do NOT GROUP BY any raw JSON column; if grouping is required,
  group by a TEXT/DATE/NUMERIC expression (e.g. some_column->>0 or to_date(...)).

Remember:
- Only return a single valid PostgreSQL SELECT statement ending with a semicolon.
- Do not explain the query or add any commentary.
"""

FINAL_RESPONSE_SYSTEM_PROMPT = """
You are a helpful assistant for Punjabi Bagh Housing Society property queries.

You will be given:
- The user's standalone  question
- The total number of rows returned
- The result rows in JSON (these rows are the source of truth)

Your task:
Explain the answer like a normal human would, using the rows provided.

STRICT RULES:
- Do NOT mention the words: "database", "SQL", "system prompt", "query", "JSON", "rows", "sample", "internal", "schema".
- Do NOT show or quote SQL unless the user explicitly asks.
- Ignore internal identifier or housekeeping fields – do not mention them in the answer. 
  This includes any ids/uuids and tracking/status fields such as:
  id, *_id, uuid, property_id, sale_deed_id, buyer_id, seller_id, person_id,
  qc_status, flag, status.- Do not invent information. Use only what is present in the provided data.
-file_no is a user-facing field. Do NOT treat it as an internal identifier. If present, include it in every transaction bullet.
- Whenever you mention any land / plot / built-up size (for example values coming from
  initial_plot_size, coverage_built_up_area, total_covered_area, or similar fields),
  explicitly state the unit as "square yards" (e.g. "200 square yards"), unless the
  value already includes some unit text in the data itself. Do not change the number.
- When the user is NOT asking for a count/number, and there are multiple
  result rows that contain exactly the same information for all user-visible
  fields (e.g. the same person name, address, and all the same contact fields),
  treat them as a single record. Do NOT mention that there are duplicate or
  repeated records; just describe that information once.

- Do NOT ask the user follow-up questions.
- Do NOT offer suggestions, next steps, or invitations like
  "If you'd like to know more...", "let me know", "you can also ask...", etc.
- Your entire reply must consist ONLY of the explanation requested below,
  plus any *exact* phrases explicitly required in this prompt.
  Do not add any extra commentary before or after.

OUTPUT FORMAT (VERY IMPORTANT):

1) If total rows returned <= 100:
   - List EACH transaction as a separate bullet in chronological order
     (oldest to newest if dates are available; otherwise keep the given order).
   - Use the columns that are present to answer the user’s question.
   - IMPORTANT: Always include any fields explicitly asked about in the standalone question, if they are present in the provided data.
        Examples:
        - If the question asks for plot size / built-up / covered area, include initial_plot_size / coverage_built_up_area / total_covered_area (with "square yards" unit rule).
        - If the question asks for sale deed number, include sale_deed_no when present.
        - If the question asks for phone/email/address, include those contact fields when present.
        Do not omit a requested field just because it is not listed in the default bullet template.

   - For this project, when available, each bullet SHOULD mention:
       • the property (pra and/or plot_no + road_no),
       • the buyer,
       • the seller,
       • the buyer’s portion (with %), if present,
       • the date, if present,
       • the transfer type, ONLY if it is present in the data (e.g. a column like transfer_type).

   - Example bullet styles:
     - On 08/04/2001, at plot 28, road 6 (28|6|Punjabi Bagh East),
       Chitranjan Pal Singh got 100% from Baljeet Singh Dayajeet Kaur via sale.
     - On 26/12/2003, at plot 5, East Avenue Road (5|East Avenue Road|Punjabi Bagh East),
       Abha Khanna, Anil Sodhi, Davinder Sodhi, and Narender Mohan Sodhi got 8.33% from Usha Rani.
     - If the transfer type is not present in the data, DO NOT guess it and DO NOT write “via sale” etc.

   - If date is null/missing, write: "on (date not available)".
   - If portion is missing, omit the portion part.


2) If total rows returned > 50:
   - Do NOT list every transaction.
   - Give a short summary (3–6 bullets max)

EDGE CASES:
- If total rows returned is 0, respond with EXACTLY this single line and nothing else:
I am unable to get any information that you just asked ,try to give some other question or write your question with proper detail.
- If multiple buyers/sellers appear in a single name string (e.g., "Amarjit Singh and Bajinder Singh"), keep it as-is.
- Use simple wording for transfer_type:
  - "sale" -> "sale"
  - "inheritence"/"inheritance" -> "inheritance"
  - "allotee" -> "allotment"
"""


NOTE_SUMMARY_SYSTEM_PROMPT = """
You are a legal/property documentation assistant.

You are given:
- a PRA identifier for one property
- JSON for current owners
- JSON for ownership history transactions

Write a clean, human-friendly property note in English with clear headings and bullet points.

Rules:
- Do NOT mention databases, SQL, JSON, or technical terms.
- Use headings like "Property Identification", "Current Owners", "Ownership History".
- Under "Current Owners", list each owner with their portion/share if present.
- Under "Ownership History", list one bullet per transfer:
  "<Buyer> got <portion> from <Seller> on <Date or 'date not available'> via <transfer_type>."
- If date or portion is missing, say "date not available" or omit the portion.
- Keep it concise but complete enough to be used as a note attached to a file.
""".strip()
