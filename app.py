
import os
import re
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer
)
from reportlab.lib.units import mm
from io import BytesIO
from datetime import datetime
from statistics import mean
from flask import Flask, render_template, request, session, send_file, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image, ImageFilter, ImageEnhance, ImageOps, ImageStat
import pytesseract
import cv2
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}

app = Flask(__name__)
app.secret_key = "legal-metrology-secret-key"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

import shutil

tesseract_path = shutil.which("tesseract")

if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path


# -----------------------------
# File validation
# -----------------------------
def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


# -----------------------------
# OCR helper functions
# -----------------------------
def normalize_text(text):
    return re.sub(r"\s+", " ", text).strip()

def contains_mrp(text):
    pattern = (
        r"\b(?:mrp|m\.r\.p\.?)\s*"
        r"(?:[:\-]|\([^)]*\)|[a-zA-Z]+\)?)*\s*"
        r"(?:rs\.?|₹|inr)?\s*"
        r"\d+(?:\.\d{1,2})?"
    )

    return bool(re.search(pattern, text, re.IGNORECASE))

def contains_quantity(text):
    pattern = (
        r"\b\d+(?:\.\d+)?\s*"
        r"(?:kg|kgs|g|gm|gram|grams|mg|"
        r"l|litre|liter|litres|liters|ml|"
        r"m|cm|mm|pcs|piece|pieces|unit|units|no\.?)\b"
    )

    return bool(re.search(pattern, text, re.IGNORECASE))

def contains_date(text):
    pattern = (
        r"\b(?:"
        r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
        r"\d{1,2}[/-]\d{4}|"
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)"
        r"[a-z]*\s+\d{4}"
        r")\b"
    )
    return bool(re.search(pattern, text, re.IGNORECASE))

def contains_phone_number(text):
    cleaned_text = re.sub(r"[\s\-()]", "", text)

    pattern = r"(?:\+91)?[6-9]\d{9}"

    return bool(re.search(pattern, cleaned_text))

def contains_email(text):
    pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    return bool(re.search(pattern, text))

def contains_address(text):
    address_keywords = [
        "address",
        "road",
        "street",
        "nagar",
        "village",
        "district",
        "state",
        "india",
        "pin",
    ]

    six_digit_pin = re.search(r"\b\d{6}\b", text)

    return (
        any(keyword in text.lower() for keyword in address_keywords)
        or bool(six_digit_pin)
    )


def contains_country_of_origin(text):
    keywords = [
        "country of origin",
        "made in",
        "manufactured in",
        "product of",
    ]

    lower_text = text.lower()
    return any(keyword in lower_text for keyword in keywords)


def contains_product_name(text):
    keywords = [
        "product",
        "name",
        "soap",
        "rice",
        "oil",
        "biscuits",
        "flour",
        "sugar",
        "shampoo",
        "detergent",
        "tea",
        "coffee",
    ]

    lower_text = text.lower()
    return any(keyword in lower_text for keyword in keywords)


def contains_unit_sale_price(text):
    pattern = (
        r"(?:per\s*(?:kg|g|gram|litre|liter|l|ml|"
        r"meter|metre|cm|piece|unit|number))"
    )
    return bool(re.search(pattern, text, re.IGNORECASE))

def contains_best_before(text):
    pattern = (
        r"best\s*before|"
        r"use\s*by|"
        r"use\s*before|"
        r"expiry|"
        r"expires?|"
        r"exp\.?\s*date"
    )

    return bool(re.search(pattern, text, re.IGNORECASE))

def has_manufacturer_details(text):
    patterns = [
        r"manufactured\s*by",
        r"manufactured\s*&\s*packed\s*by",
        r"mfd\s*by",
        r"packed\s*by",
        r"imported\s*by",
    ]
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def has_address(text):
    address_words = [
        "address", "road", "street", "nagar",
        "chennai", "india", "tamil nadu",
        "pincode", "pin code"
    ]
    return any(word in text.lower() for word in address_words)


def has_country_of_origin(text):
    return bool(re.search(
        r"country\s*of\s*origin|made\s*in|product\s*of",
        text,
        re.IGNORECASE
    ))


def has_product_name(text):
    return len(text.split()) >= 3


def has_quantity(text):
    return bool(re.search(
        r"\b\d+(?:\.\d+)?\s*(g|kg|mg|ml|l|litre|liter|cm|m)\b",
        text,
        re.IGNORECASE
    ))


def has_date(text):
    return bool(re.search(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
        r"|\b\d{1,2}[/-]\d{2,4}\b",
        text
    ))


def has_mrp(text):
    return bool(re.search(
        r"(mrp|m\.r\.p\.?)\s*[:\-]?\s*(₹|rs\.?|inr)?\s*\d+",
        text,
        re.IGNORECASE
    ))


def has_tax_inclusive_statement(text):
    return bool(re.search(
        r"incl\.?\s*of\s*all\s*taxes|inclusive\s*of\s*all\s*taxes",
        text,
        re.IGNORECASE
    ))


def has_phone_number(text):
    return bool(re.search(r"\b[6-9]\d{9}\b", text))


def has_email(text):
    return bool(re.search(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
        text
    ))


def has_unit_sale_price(text):
    return bool(re.search(
        r"(unit\s*sale\s*price|price\s*per\s*(kg|g|l|ml))",
        text,
        re.IGNORECASE
    ))


def has_best_before(text):
    return bool(re.search(
        r"best\s*before|use\s*before|expiry|exp\.?",
        text,
        re.IGNORECASE
    ))

def needs_manual_review(image, ocr_text):
    """
    Returns True when the image/text is not clear enough
    for reliable automatic OCR checking.
    """

    # Check image sharpness
    gray_image = image.convert("L")
    blurred_image = gray_image.filter(ImageFilter.GaussianBlur(radius=2))

    original_stats = ImageStat.Stat(gray_image)
    blurred_stats = ImageStat.Stat(blurred_image)

    sharpness_difference = (
        original_stats.var[0] - blurred_stats.var[0]
    )

    # Check OCR output
    readable_text = ocr_text.strip()

    # Very little readable text
    if len(readable_text) < 10:
        return True

    # Very low image sharpness
    if sharpness_difference < 20:
        return True

    return False

# -----------------------------
# Rule-engine checklist
# -----------------------------

def keyword_check(*keywords):
    def check(text):
        lower_text = text.lower()
        return any(keyword.lower() in lower_text for keyword in keywords)

    check.keywords = keywords
    check.check_type = "keyword"

    return check

def regex_check(pattern):
    def check(text):
        return bool(re.search(pattern, text, re.IGNORECASE))

    check.pattern = pattern
    check.check_type = "regex"

    return check

    def get_rule_confidence(rule, ocr_data):

        check_function = rule["check"]
        rule_id = rule.get("id", "")

    # -----------------------------------------
    # Collect OCR words and confidence
    # -----------------------------------------

    words = []

    for i, text in enumerate(ocr_data["text"]):

        text = text.strip()

        if not text:
            continue

        try:
            confidence = float(ocr_data["conf"][i])
        except:
            confidence = 0

        words.append({
            "text": text.lower(),
            "confidence": confidence
        })

    if not words:
        return None

    # -----------------------------------------
    # Create complete OCR text
    # -----------------------------------------

    full_text = " ".join(
        word["text"]
        for word in words
    )

    confidence_values = []

    # -----------------------------------------
    # Keyword rules
    # -----------------------------------------

    check_type = check_function.__dict__.get(
        "check_type"
    )

    if check_type == "keyword":

        keywords = check_function.__dict__.get(
            "keywords",
            []
        )

        for keyword in keywords:

            keyword = keyword.lower().strip()

            if not keyword:
                continue

            # ---------------------------------
            # Normalize keyword
            # ---------------------------------

            keyword_normalized = re.sub(
                r"[^a-z0-9₹@.+-]",
                " ",
                keyword
            )

            keyword_normalized = re.sub(
                r"\s+",
                " ",
                keyword_normalized
            ).strip()

            # ---------------------------------
            # Search complete OCR text
            # ---------------------------------

            if keyword_normalized in full_text:

                keyword_parts = keyword_normalized.split()

                # Find matching OCR words
                for start in range(len(words)):

                    candidate = words[
                        start:start + len(keyword_parts)
                    ]

                    if len(candidate) != len(keyword_parts):
                        continue

                    candidate_text = " ".join(
                        item["text"]
                        for item in candidate
                    )

                    candidate_text = re.sub(
                        r"\s+",
                        " ",
                        candidate_text
                    ).strip()

                    if (
                        candidate_text == keyword_normalized
                        or keyword_normalized in candidate_text
                        or candidate_text in keyword_normalized
                    ):

                        confidence_values.extend(
                            item["confidence"]
                            for item in candidate
                        )

            # ---------------------------------
            # Single-word / OCR variation match
            # ---------------------------------

            else:

                for word in words:

                    word_text = word["text"]

                    if (
                        word_text == keyword_normalized
                        or keyword_normalized in word_text
                        or word_text in keyword_normalized
                    ):

                        confidence_values.append(
                            word["confidence"]
                        )

    # -----------------------------------------
    # Regex rules
    # -----------------------------------------

    elif check_type == "regex":

        pattern = check_function.__dict__.get(
            "pattern"
        )

        if pattern:

            # Search complete OCR text first
            try:

                if re.search(
                    pattern,
                    full_text,
                    re.IGNORECASE
                ):

                    # Collect confidence of
                    # relevant OCR words

                    for word in words:

                        try:

                            if re.search(
                                pattern,
                                word["text"],
                                re.IGNORECASE
                            ):
                                confidence_values.append(
                                    word["confidence"]
                                )

                        except:
                            pass

            except:
                pass

    # -----------------------------------------
    # Custom contains_* functions
    # -----------------------------------------

    else:

        function_name = check_function.__name__

        related_words = []

        if function_name == "contains_mrp":

            related_words = [
                "mrp",
                "m.r.p",
                "price",
                "₹",
                "rs",
                "inr"
            ]

        elif function_name == "contains_quantity":

            related_words = [
                "quantity",
                "net",
                "kg",
                "kgs",
                "g",
                "gm",
                "gram",
                "grams",
                "mg",
                "ml",
                "litre",
                "liter",
                "litres",
                "liters",
                "l",
                "pcs",
                "piece",
                "pieces",
                "unit",
                "units"
            ]

        elif function_name == "contains_phone_number":

            related_words = [
                "phone",
                "mobile",
                "contact",
                "+91"
            ]

        elif function_name == "contains_email":

            related_words = [
                "@",
                "email",
                "care",
                ".com",
                ".in"
            ]

        elif function_name == "contains_best_before":

            related_words = [
                "best",
                "before",
                "use",
                "by",
                "expiry",
                "expires",
                "exp"
            ]

        elif function_name == "contains_unit_sale_price":

            related_words = [
                "unit",
                "sale",
                "price",
                "per",
                "kg",
                "g",
                "litre",
                "liter",
                "ml",
                "piece",
                "number"
            ]

        # -------------------------------------
        # Search related OCR words
        # -------------------------------------

        for word in words:

            word_text = word["text"]

            for related_word in related_words:

                related_word = related_word.lower()

                # Avoid false matching
                # with one-letter units

                if len(related_word) == 1:

                    if word_text == related_word:

                        confidence_values.append(
                            word["confidence"]
                        )

                        break

                else:

                    if (
                        word_text == related_word
                        or related_word in word_text
                    ):

                        confidence_values.append(
                            word["confidence"]
                        )

                        break

    # -----------------------------------------
    # Rule-specific confidence words
    # -----------------------------------------

    if not confidence_values:

        rule_words = {

            "R01": [
                "manufactured",
                "packed",
                "imported",
                "manufacturer",
                "packer",
                "importer",
                "mfd"
            ],

            "R02": [
                "address",
                "road",
                "street",
                "nagar",
                "village",
                "district",
                "state",
                "india",
                "tamil",
                "nadu",
                "industrial",
                "area",
                "pincode",
                "pin"
            ],

            "R03": [
                "country",
                "origin",
                "made",
                "manufactured",
                "product",
                "india"
            ],

            "R04": [
                "soap",
                "rice",
                "oil",
                "biscuits",
                "flour",
                "sugar",
                "shampoo",
                "detergent",
                "tea",
                "coffee",
                "turmeric",
                "powder",
                "haldi"
            ],

            "R14": [
                "declaration",
                "warning",
                "instruction",
                "consumer",
                "caution",
                "storage",
                "information"
            ],

            "R16": [
                "inner",
                "package",
                "outer",
                "net",
                "quantity",
                "mrp"
            ],

            "R17": [
                "net",
                "quantity",
                "standard",
                "kg",
                "g",
                "ml",
                "litre",
                "liter"
            ],

            "R18": [
                "net",
                "quantity",
                "mrp",
                "manufactured",
                "packed"
            ],

            "R19": [
                "inspection",
                "inspected",
                "testing",
                "tested"
            ],

            "R20": [
                "sample",
                "sampling"
            ],

            "R21": [
                "test",
                "procedure",
                "testing",
                "laboratory"
            ],

            "R22": [
                "verification",
                "verified",
                "mark"
            ],

            "R23": [
                "registered",
                "manufacturer",
                "registration",
                "number"
            ],

            "R24": [
                "registered",
                "packer",
                "registration",
                "number"
            ],

            "R25": [
                "registered",
                "importer",
                "registration",
                "number"
            ],

            "R26": [
                "registered",
                "manufacturers",
                "packers",
                "registration",
                "list"
            ],

            "R27": [
                "net",
                "quantity",
                "declaration"
            ],

            "R28": [
                "font",
                "size",
                "letter",
                "minimum"
            ],

            "R29": [
                "penalty",
                "contravention",
                "offence"
            ],

            "R30": [
                "penalty",
                "contravention",
                "offence",
                "other"
            ],

            "R31": [
                "relaxation",
                "relax",
                "provisions",
                "government",
                "exemption"
            ],

            "R32": [
                "repeal",
                "repealed",
                "previous",
                "rules"
            ],

            "R33": [
                "saving",
                "savings",
                "actions",
                "proceedings"
            ],

            "R34": [
                "existing",
                "registration",
                "proceedings",
                "continued"
            ]
        }

        related_words = rule_words.get(
            rule_id,
            []
        )

        for word in words:

            word_text = word["text"]

            for related_word in related_words:

                related_word = related_word.lower()

                if len(related_word) == 1:

                    if word_text == related_word:

                        confidence_values.append(
                            word["confidence"]
                        )

                        break

                else:

                    if (
                        word_text == related_word
                        or related_word in word_text
                    ):

                        confidence_values.append(
                            word["confidence"]
                        )

                        break

    # -----------------------------------------
    # No confidence evidence
    # -----------------------------------------

    if not confidence_values:
        return None

    # -----------------------------------------
    # Final confidence
    # -----------------------------------------

    return mean(confidence_values)

def get_rule_confidence(rule, ocr_data):

    check_function = rule["check"]
    rule_id = rule.get("id", "")

    # ---------------------------------
    # Collect OCR words and confidence
    # ---------------------------------

    words = []

    if not ocr_data:
        return None

    ocr_words = ocr_data.get("text", [])
    ocr_confidences = ocr_data.get("conf", [])

    for i, text in enumerate(ocr_words):

        text = str(text).strip()

        if not text:
            continue

        try:
            confidence = float(
                ocr_confidences[i]
            )
        except Exception:
            confidence = 0

        if confidence < 0:
            confidence = 0

        words.append({
            "text": text.lower(),
            "confidence": confidence
        })

    if not words:
        return None

    confidence_values = []

    # ---------------------------------
    # 1. Keyword-based rules
    # ---------------------------------

    check_type = getattr(
        check_function,
        "check_type",
        None
    )

    if check_type == "keyword":

        keywords = getattr(
            check_function,
            "keywords",
            []
        )

        for keyword in keywords:

            keyword = str(
                keyword
            ).lower().strip()

            if not keyword:
                continue

            keyword_parts = keyword.split()

            # Single-word keyword
            if len(keyword_parts) == 1:

                for word in words:

                    if (
                        keyword in word["text"]
                        or word["text"] in keyword
                    ):
                        confidence_values.append(
                            word["confidence"]
                        )

            # Multi-word keyword
            else:

                for i in range(
                    len(words) - len(keyword_parts) + 1
                ):

                    sequence = words[
                        i:i + len(keyword_parts)
                    ]

                    sequence_text = " ".join(
                        item["text"]
                        for item in sequence
                    )

                    if keyword in sequence_text:

                        confidence_values.extend(
                            item["confidence"]
                            for item in sequence
                        )

    # ---------------------------------
    # 2. Regex-based rules
    # ---------------------------------

    elif check_type == "regex":

        pattern = getattr(
            check_function,
            "pattern",
            None
        )

        if pattern:

            full_text = " ".join(
                word["text"]
                for word in words
            )

            try:

                matches = list(
                    re.finditer(
                        pattern,
                        full_text,
                        re.IGNORECASE
                    )
                )

                if matches:

                    for match in matches:

                        matched_text = (
                            match.group(0)
                            .lower()
                        )

                        matched_parts = (
                            matched_text.split()
                        )

                        for part in matched_parts:

                            for word in words:

                                if (
                                    part in word["text"]
                                    or
                                    word["text"] in part
                                ):
                                    confidence_values.append(
                                        word["confidence"]
                                    )
                                    break

            except Exception:
                pass

    # ---------------------------------
    # 3. Special rule-related words
    # ---------------------------------

    related_words = []

    if rule_id == "R01":

        related_words = [
            "manufacturer",
            "manufactured",
            "manufactured by",
            "packer",
            "packed",
            "packed by",
            "importer",
            "imported"
        ]

    elif rule_id == "R02":

        related_words = [
            "address",
            "road",
            "street",
            "nagar",
            "village",
            "district",
            "state",
            "industrial",
            "area",
            "india"
        ]

    elif rule_id == "R03":

        related_words = [
            "country",
            "origin",
            "country of origin",
            "made in",
            "product of",
            "manufactured in"
        ]

    elif rule_id == "R04":

        related_words = [
            "product",
            "name",
            "turmeric",
            "powder",
            "soap",
            "rice",
            "oil",
            "biscuits",
            "flour",
            "sugar",
            "shampoo",
            "detergent",
            "tea",
            "coffee"
        ]

    elif rule_id == "R05":

        related_words = [
            "quantity",
            "net quantity",
            "kg",
            "g",
            "gm",
            "mg",
            "ml",
            "litre",
            "liter"
        ]

    elif rule_id == "R06":

        related_words = [
            "kg",
            "g",
            "gm",
            "mg",
            "ml",
            "litre",
            "liter",
            "unit",
            "pcs"
        ]

    elif rule_id == "R07":

        related_words = [
            "manufactured",
            "manufacturing",
            "packed",
            "packed on",
            "date",
            "mfd"
        ]

    elif rule_id == "R08":

        related_words = [
            "mrp",
            "maximum",
            "retail",
            "price",
            "rs"
        ]

    elif rule_id == "R09":

        related_words = [
            "inclusive",
            "taxes",
            "tax",
            "all taxes",
            "inclusive of all taxes"
        ]

    elif rule_id == "R10":

        related_words = [
            "phone",
            "telephone",
            "contact",
            "customer care"
        ]

    elif rule_id == "R11":

        related_words = [
            "email",
            "mail",
            "@"
        ]

    elif rule_id == "R12":

        related_words = [
            "unit",
            "sale",
            "price",
            "per kg",
            "per g",
            "per litre",
            "per ml"
        ]

    elif rule_id == "R13":

        related_words = [
            "best",
            "before",
            "expiry",
            "expiry date",
            "use before"
        ]

    elif rule_id == "R14":

        related_words = [
            "warning",
            "instruction",
            "declaration",
            "caution"
        ]

    elif rule_id == "R15":

        related_words = [
            "quantity",
            "net quantity"
        ]

    elif rule_id == "R16":

        related_words = [
            "inner package",
            "outer package",
            "net quantity"
        ]

    elif rule_id == "R17":

        related_words = [
            "quantity",
            "net quantity"
        ]

    elif rule_id == "R18":

        related_words = [
            "net quantity",
            "mrp",
            "manufactured",
            "packed"
        ]

    elif rule_id == "R19":

        related_words = [
            "inspection",
            "inspected",
            "testing",
            "tested"
        ]

    elif rule_id == "R20":

        related_words = [
            "sample",
            "sampling"
        ]

    elif rule_id == "R21":

        related_words = [
            "test procedure",
            "testing procedure",
            "laboratory"
        ]

    elif rule_id == "R22":

        related_words = [
            "verification",
            "verified",
            "verification mark"
        ]

    elif rule_id == "R23":

        related_words = [
            "registration number",
            "registered manufacturer"
        ]

    elif rule_id == "R24":

        related_words = [
            "registration number",
            "registered packer"
        ]

    elif rule_id == "R25":

        related_words = [
            "registration number",
            "registered importer"
        ]

    elif rule_id == "R26":

        related_words = [
            "registered manufacturer",
            "registered packer",
            "registration number"
        ]

    elif rule_id == "R27":

        related_words = [
            "net quantity",
            "quantity declaration"
        ]

    elif rule_id == "R28":

        related_words = [
            "font size",
            "letter size",
            "minimum size"
        ]

    elif rule_id == "R29":

        related_words = [
            "penalty",
            "contravention",
            "offence"
        ]

    elif rule_id == "R30":

        related_words = [
            "penalty",
            "contravention",
            "offence",
            "other"
        ]

    elif rule_id == "R31":

        related_words = [
            "relaxation",
            "relax",
            "provisions",
            "government",
            "exemption"
        ]

    elif rule_id == "R32":

        related_words = [
            "repeal",
            "repealed",
            "previous",
            "rules"
        ]

    elif rule_id == "R33":

        related_words = [
            "saving",
            "savings",
            "actions",
            "proceedings"
        ]

    elif rule_id == "R34":

        related_words = [
            "existing",
            "registration",
            "proceedings",
            "continued"
        ]

    # ---------------------------------
    # 4. Find confidence from
    #    rule-specific words
    # ---------------------------------

    if not confidence_values and related_words:

        for word in words:

            word_text = word["text"]

            for related_word in related_words:

                related_word = (
                    related_word.lower()
                )

                if (
                    related_word in word_text
                    or word_text in related_word
                ):

                    confidence_values.append(
                        word["confidence"]
                    )

                    break

    # ---------------------------------
    # 5. No matching OCR information
    # ---------------------------------

    if not confidence_values:
        return None

    # ---------------------------------
    # 6. Final confidence
    # ---------------------------------

    return mean(confidence_values)

def supplementary_rule_check(rule, ocr_text):

    text = normalize_text(ocr_text).lower()

    # ==========================================
    # COMMON OCR NORMALIZATION
    # ==========================================

    text = re.sub(
        r"m\s*\.?\s*r\s*\.?\s*p\s*\.?",
        "mrp",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"e\s*[-]?\s*mail",
        "email",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"batch\s*no\.?",
        "batch no",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"best\s+before",
        "best before",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"country\s+of\s+origin",
        "country of origin",
        text,
        flags=re.IGNORECASE
    )

    rule_id = rule.get("id", "")


    # ==========================================
    # R01 - MANUFACTURER / PACKER / IMPORTER
    # ==========================================

    if rule_id == "R01":

        patterns = [

            r"\bmanufactured\s+by\b",

            r"\bmanufactured\s*&?\s*packed\s+by\b",

            r"\bmanufactured\s+and\s+packed\s+by\b",

            r"\bpacked\s+by\b",

            r"\bimported\s+by\b",

            r"\bimporter\b",

            r"\bmanufacturer\b",

            r"\bpacker\b",

            r"\bmfd\s+by\b",

            r"\bmfg\s+by\b",

            r"\bmfd\s*&\s*\w+\s+by\b",

            r"\bmfd\b.{0,30}\bby\b"
        ]

        return any(
            re.search(
                pattern,
                text,
                re.IGNORECASE
            )
            for pattern in patterns
        )


    # ==========================================
    # R02 - COMPLETE ADDRESS
    # ==========================================

    if rule_id == "R02":

        address_patterns = [

            r"\baddress\b",

            r"\broad\b",

            r"\bstreet\b",

            r"\bnagar\b",

            r"\bvillage\b",

            r"\bdistrict\b",

            r"\bstate\b",

            r"\btamil\s+nadu\b",

            r"\bindia\b",

            r"\bpincode\b",

            r"\bpin\s*code\b",

            r"\b\d{6}\b"
        ]

        matches = sum(
            bool(
                re.search(
                    pattern,
                    text,
                    re.IGNORECASE
                )
            )
            for pattern in address_patterns
        )

        return matches >= 2


    # ==========================================
    # R03 - COUNTRY OF ORIGIN
    # ==========================================

    if rule_id == "R03":

        patterns = [

            r"\bcountry\s+of\s+origin\b",

            r"\bmade\s+in\s+[a-zA-Z]+\b",

            r"\bmanufactured\s+in\s+[a-zA-Z]+\b",

            r"\bproduct\s+of\s+[a-zA-Z]+\b"
        ]

        return any(
            re.search(
                pattern,
                text,
                re.IGNORECASE
            )
            for pattern in patterns
        )


    # ==========================================
    # R04 - COMMON / GENERIC PRODUCT NAME
    # ==========================================

    if rule_id == "R04":

        product_patterns = [

            r"\bsoap\b",

            r"\brice\b",

            r"\boil\b",

            r"\bbiscuits?\b",

            r"\bflour\b",

            r"\bsugar\b",

            r"\bshampoo\b",

            r"\bdetergent\b",

            r"\btea\b",

            r"\bcoffee\b",

            r"\bturmeric\b",

            r"\bturmeric\s+powder\b",

            r"\bhaldi\b"
        ]

        return any(
            re.search(
                pattern,
                text,
                re.IGNORECASE
            )
            for pattern in product_patterns
        )


    # ==========================================
    # R05 - NET QUANTITY
    # ==========================================

    if rule_id == "R05":

        quantity_pattern = (
            r"\b\d+(?:\.\d+)?\s*"
            r"(?:kg|kgs|g|gm|gram|grams|mg|"
            r"ml|l|litre|liter|litres|liters|"
            r"pcs|pieces|unit|units)\b"
        )

        return bool(
            re.search(
                quantity_pattern,
                text,
                re.IGNORECASE
            )
        )


    # ==========================================
    # R06 - QUANTITY UNIT
    # ==========================================

    if rule_id == "R06":

        quantity_unit_pattern = (
            r"\b\d+(?:\.\d+)?\s*"
            r"(?:kg|kgs|g|gm|gram|grams|mg|"
            r"ml|l|litre|liter|litres|liters|"
            r"pcs|pieces|unit|units)\b"
        )

        return bool(
            re.search(
                quantity_unit_pattern,
                text,
                re.IGNORECASE
            )
        )


    # ==========================================
    # R07 - MANUFACTURING / PACKING DATE
    # ==========================================

    if rule_id == "R07":

        date_pattern = (
            r"\b\d{1,2}"
            r"[-/]\d{1,2}"
            r"[-/]\d{2,4}\b"
        )

        manufacturing_words = [
            "mfd",
            "mfg",
            "manufactured",
            "manufacturing",
            "packed",
            "packing"
        ]

        has_date = bool(
            re.search(
                date_pattern,
                text
            )
        )

        has_manufacturing_word = any(
            word in text
            for word in manufacturing_words
        )

        return (
            has_date
            and has_manufacturing_word
        )


    # ==========================================
    # R08 - MRP
    # ==========================================

    if rule_id == "R08":

        patterns = [

            # MRP followed by price
            r"\bmrp\b.{0,50}"
            r"(?:rs\.?|inr|₹)?\s*"
            r"\d+(?:\.\d{1,2})?",

            # MRP with Rs
            r"\bmrp\b.{0,50}"
            r"\brs\.?\s*"
            r"\d+(?:\.\d{1,2})?",

            # Maximum Retail Price
            r"\bmaximum\s+retail\s+price\b",

            # M.R.P
            r"\bm\.r\.p\b.{0,50}"
            r"\d+(?:\.\d{1,2})?"
        ]

        return any(
            re.search(
                pattern,
                text,
                re.IGNORECASE
            )
            for pattern in patterns
        )


    # ==========================================
    # R09 - MRP INCLUSIVE OF TAXES
    # ==========================================

    if rule_id == "R09":

        patterns = [

            r"inclusive\s+of\s+all\s+taxes",

            r"inclusive\s+of\s+tax",

            r"all\s+taxes\s+included",

            r"inclusive.*taxes"
        ]

        return any(
            re.search(
                pattern,
                text,
                re.IGNORECASE
            )
            for pattern in patterns
        )


    # ==========================================
    # R10 - CONSUMER CARE PHONE
    # ==========================================

    if rule_id == "R10":

        phone_pattern = (
            r"(?:\+91[\s-]?)?"
            r"[6-9]\d{4}"
            r"[\s-]?\d{5}"
        )

        return bool(
            re.search(
                phone_pattern,
                text
            )
        )


    # ==========================================
    # R11 - CONSUMER CARE EMAIL
    # ==========================================

    if rule_id == "R11":

        email_pattern = (
            r"[A-Za-z0-9._%+-]+"
            r"@[A-Za-z0-9.-]+"
            r"\.[A-Za-z]{2,}"
        )

        return bool(
            re.search(
                email_pattern,
                text
            )
        )


    # ==========================================
    # R12 - UNIT SALE PRICE
    # ==========================================

    if rule_id == "R12":

        patterns = [

            r"\bunit\s+sale\s+price\b",

            r"\bsale\s+price\b",

            r"\bper\s+kg\b",

            r"\bper\s+g\b",

            r"\bper\s+gram\b",

            r"\bper\s+ml\b",

            r"\bper\s+litre\b",

            r"\bper\s+liter\b",

            r"\bper\s+piece\b",

            r"\bper\s+unit\b"
        ]

        return any(
            re.search(
                pattern,
                text,
                re.IGNORECASE
            )
            for pattern in patterns
        )


    # ==========================================
    # R13 - BEST BEFORE / USE BY
    # ==========================================

    if rule_id == "R13":

        patterns = [

            r"\bbest\s+before\b",

            r"\buse\s+by\b",

            r"\buse\s+before\b",

            r"\bexpiry\b",

            r"\bexpires?\b",

            r"\bexp\.?\s*date\b"
        ]

        return any(
            re.search(
                pattern,
                text,
                re.IGNORECASE
            )
            for pattern in patterns
        )


    # ==========================================
    # R14 - VISIBILITY / LEGIBILITY
    # ==========================================

    if rule_id == "R14":

        important_declarations = [

            "mrp",

            "net quantity",

            "manufactured",

            "packed",

            "best before",

            "consumer care"
        ]

        matches = sum(
            declaration in text
            for declaration
            in important_declarations
        )

        return matches >= 2


    # ==========================================
    # R15 - NON-MISLEADING QUANTITY
    # ==========================================

    if rule_id == "R15":

        quantity_pattern = (
            r"\b\d+(?:\.\d+)?\s*"
            r"(?:kg|kgs|g|gm|gram|grams|mg|"
            r"ml|l|litre|liter|litres|liters|"
            r"pcs|pieces|unit|units)\b"
        )

        return bool(
            re.search(
                quantity_pattern,
                text,
                re.IGNORECASE
            )
        )


    # ==========================================
    # R16 - INNER / OUTER PACKAGE
    # ==========================================

    if rule_id == "R16":

        patterns = [

            r"\binner\s+package\b",

            r"\bouter\s+package\b",

            r"\binner\b.{0,30}\bpackage\b",

            r"\bouter\b.{0,30}\bpackage\b"
        ]

        return any(
            re.search(
                pattern,
                text,
                re.IGNORECASE
            )
            for pattern in patterns
        )


    # ==========================================
    # R17 - QUANTITY ACCURACY
    # ==========================================

    if rule_id == "R17":

        quantity_pattern = (
            r"\b\d+(?:\.\d+)?\s*"
            r"(?:kg|kgs|g|gm|gram|grams|mg|"
            r"ml|l|litre|liter|litres|liters|"
            r"pcs|pieces|unit|units)\b"
        )

        return bool(
            re.search(
                quantity_pattern,
                text,
                re.IGNORECASE
            )
        )


    # ==========================================
    # R18 - MANDATORY PACKAGE DECLARATIONS
    # ==========================================

    if rule_id == "R18":

        important_declarations = [

            "net quantity",
            "mrp",
            "manufactured",
            "packed",
            "manufactured by",
            "packed by"
        ]

        matches = sum(
            declaration in text
            for declaration
            in important_declarations
        )

        return matches >= 2


    # ==========================================
    # R19-R26
    # EXTERNAL / MANUAL VERIFICATION
    # ==========================================

    if rule_id in [
        "R19",
        "R20",
        "R21",
        "R22",
        "R23",
        "R24",
        "R25",
        "R26"
    ]:
        return False


    # ==========================================
    # R27 - QUANTITY DECLARATION
    # ==========================================

    if rule_id == "R27":

        return (
            "net quantity" in text
            or "quantity" in text
        )


    # ==========================================
    # R28 - FONT / LETTER SIZE
    # ==========================================

    if rule_id == "R28":

        patterns = [

            r"\bfont\b",

            r"\bfont\s+size\b",

            r"\bletter\s+size\b",

            r"\bminimum\s+size\b"
        ]

        return any(
            re.search(
                pattern,
                text,
                re.IGNORECASE
            )
            for pattern in patterns
        )


    # ==========================================
    # R29-R34
    # NOT ESTABLISHED FROM PACKAGE OCR
    # ==========================================

    if rule_id in [
        "R29",
        "R30",
        "R31",
        "R32",
        "R33",
        "R34"
    ]:
        return False


    return False

COMPLIANCE_RULES = [
    {
        "id": "R01",
        "field": "Manufacturer / Packer / Importer Name",
        "rule": "Rule 6(1)(a)",
        "check": keyword_check(
            "manufactured by",
            "manufactured & packed by",
            "mfd by",
            "packed by",
            "imported by",
            "importer"
        ),
        "description": "Checks manufacturer, packer, or importer information."
    },
    {
        "id": "R02",
        "field": "Complete Address",
        "rule": "Rule 6(1)(a), Rule 10",
        "check": keyword_check(
            "address",
            "road",
            "street",
            "nagar",
            "village",
            "district",
            "state",
            "pincode",
            "pin code"
        ),
        "description": "Checks address-related information."
    },
    {
        "id": "R03",
        "field": "Country of Origin",
        "rule": "Rule 6(1)(aa)",
        "check": keyword_check(
            "country of origin",
            "made in",
            "manufactured in",
            "product of"
        ),
        "description": "Checks country-of-origin information."
    },
    {
        "id": "R04",
        "field": "Common / Generic Product Name",
        "rule": "Rule 6(1)(b)",
        "check": keyword_check(
            "soap",
            "rice",
            "oil",
            "biscuits",
            "flour",
            "sugar",
            "shampoo",
            "detergent",
            "tea",
            "coffee",
            "turmeric powder",
            "haldi"
        ),
        "description": "Checks for product-name information."
    },
    {
        "id": "R05",
        "field": "Net Quantity",
        "rule": "Rule 6(1)(f)",
        "check": regex_check(
            r"\b\d+(?:\.\d+)?\s*(?:kg|g|gm|gram|grams|mg|l|litre|liter|ml|m|cm|mm|pcs|pieces|units?)\b"
        ),
        "description": "Checks for numerical quantity information."
    },
    {
        "id": "R06",
        "field": "Quantity Unit",
        "rule": "Rule 6(1)(f)",
        "check": regex_check(
            r"\b\d+(?:\.\d+)?\s*(?:g|kg|mg|ml|l|litre|liter|cm|m)\b"
        ),
        "description": "Checks for quantity measurement units."
    },
    {
        "id": "R07",
        "field": "Manufacturing / Packing Date",
        "rule": "Rule 6(1)(d)",
        "check": regex_check(
            r"manufactur(?:ed|ing)|mfg|packed on|packing date|date of packing|"
            r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|"
            r"\b\d{1,2}[/-]\d{2,4}\b"
        ),
        "description": "Checks for manufacturing, packing, or date information."
    },
    {
        "id": "R08",
        "field": "Maximum Retail Price",
        "rule": "Rule 6(1)(e)",
        "check": contains_mrp,
        "description": "Checks for MRP information."
    },
    {
        "id": "R09",
        "field": "MRP Inclusive of All Taxes",
        "rule": "Rule 6(1)(e)",
        "check": regex_check(
            r"(mrp|m\.r\.p\.?).{0,50}"
            r"(inclusive|incl\.?|all taxes)"
        ),
        "description": "Checks for MRP inclusive-of-tax wording."
    },
    {
        "id": "R10",
        "field": "Consumer-care Telephone Number",
        "rule": "Rule 6(2)",
        "check": contains_phone_number,
        "description": "Checks for a consumer-care telephone number."
    },
    {
        "id": "R11",
        "field": "Consumer-care Email Address",
        "rule": "Rule 6(2)",
        "check": contains_email,
        "description": "Checks for a consumer-care email address."
    },
    {
        "id": "R12",
        "field": "Unit Sale Price",
        "rule": "Rule 6(11)",
        "check": contains_unit_sale_price,
        "description": "Checks for unit sale price information."
    },
    {
        "id": "R13",
        "field": "Best Before / Use By Declaration",
        "rule": "Rule 6(1)(da)",
        "check": contains_best_before,
        "description": "Checks for best-before or use-by information."
    },
    {
        "id": "R14",
        "field": "Visibility and Legibility of Declarations",
        "rule": "Rule 7 and Rule 10",
        "check": keyword_check(
            "mrp",
            "net quantity",
            "manufactured",
            "packed",
            "consumer care",
            "best before"
        ),
        "description": "Checks whether important declarations appear in OCR text."
    },
    {
    "id": "R15",
    "field": "Non-misleading Quantity Declaration",
    "rule": "Rule 12(6)",
    "check": contains_quantity,
    "description": "Checks whether a quantity declaration is present and expressed with a recognized unit."
},
{
    "id": "R16",
    "field": "Inner / Outer Package Declarations",
    "rule": "Rule 9(3)",
    "check": keyword_check(
        "inner package",
        "outer package",
        "net quantity"
    ),
    "description": "Checks for declarations related to inner or outer packaging."
},
{
    "id": "R17",
    "field": "Quantity Accuracy",
    "rule": "Rule 19",
    "check": contains_quantity,
    "description": "Checks whether a declared quantity is available for further verification."
},
{
    "id": "R18",
    "field": "Mandatory Declarations on Package",
    "rule": "Rule 19",
    "check": keyword_check(
        "net quantity",
        "mrp",
        "manufactured",
        "packed",
        "manufactured by",
        "packed by"
    ),
    "description": "Checks for important package declaration information."
},
{
    "id": "R19",
    "field": "Inspection and Testing",
    "rule": "Manual Verification",
    "check": keyword_check(
        "inspection",
        "inspected",
        "testing",
        "tested"
    ),
    "description": "Requires manual verification because inspection or testing cannot be established reliably from package OCR alone."
},
{
    "id": "R20",
    "field": "Sampling Requirements",
    "rule": "Manual Verification",
    "check": keyword_check(
        "sample",
        "sampling"
    ),
    "description": "Sampling activity cannot be verified from a package image alone."
},
{
    "id": "R21",
    "field": "Testing Procedure",
    "rule": "Manual Verification",
    "check": keyword_check(
        "test procedure",
        "testing procedure",
        "laboratory"
    ),
    "description": "Laboratory testing and procedures require external records or manual verification."
},
{
    "id": "R22",
    "field": "Package Verification",
    "rule": "Manual Verification",
    "check": keyword_check(
        "verification",
        "verified",
        "verification mark"
    ),
    "description": "Package verification cannot be established solely from OCR text."
},
{
    "id": "R23",
    "field": "Manufacturer Registration",
    "rule": "Manual Verification",
    "check": keyword_check(
        "registration number",
        "registered manufacturer"
    ),
    "description": "Registration status requires verification against registration records."
},
{
    "id": "R24",
    "field": "Packer Registration",
    "rule": "Manual Verification",
    "check": keyword_check(
        "registration number",
        "registered packer"
    ),
    "description": "Packer registration status requires verification against registration records."
},
{
    "id": "R25",
    "field": "Importer Registration",
    "rule": "Manual Verification",
    "check": keyword_check(
        "registration number",
        "registered importer"
    ),
    "description": "Importer registration status requires verification against registration records."
},
{
    "id": "R26",
    "field": "Registered Manufacturer / Packer Records",
    "rule": "Manual Verification",
    "check": keyword_check(
        "registered manufacturer",
        "registered packer",
        "registration number"
    ),
    "description": "Registration records cannot be verified from package OCR alone."
},
{
    "id": "R27",
    "field": "Advertisement Quantity Declaration",
    "rule": "Manual Verification",
    "check": keyword_check(
        "net quantity",
        "quantity declaration"
    ),
    "description": "Advertisement-related compliance requires the advertisement itself for verification."
},
{
    "id": "R28",
    "field": "Advertisement Font Size",
    "rule": "Manual Verification",
    "check": keyword_check(
        "font size",
        "letter size",
        "minimum size"
    ),
    "description": "Font-size compliance requires visual measurement and cannot be reliably verified from OCR text alone."
},
{
    "id": "R29",
    "field": "Penalty / Contravention",
    "rule": "Manual Verification",
    "check": keyword_check(
        "penalty",
        "contravention",
        "offence"
    ),
    "description": "Penalty or offence status is an enforcement matter and cannot be determined from package OCR."
},
{
    "id": "R30",
    "field": "Other Contraventions",
    "rule": "Manual Verification",
    "check": keyword_check(
        "penalty",
        "contravention",
        "offence"
    ),
    "description": "Contravention status requires inspection or enforcement records."
},
{
    "id": "R31",
    "field": "Government Relaxation / Exemption",
    "rule": "Manual Verification",
    "check": keyword_check(
        "relaxation",
        "relax the provisions",
        "government",
        "exemption"
    ),
    "description": "Applicable government relaxation or exemption cannot be determined from package OCR alone."
},
{
    "id": "R32",
    "field": "Repeal / Previous Rules",
    "rule": "Manual Verification",
    "check": keyword_check(
        "repeal",
        "repealed",
        "previous rules"
    ),
    "description": "Legal applicability of repealed provisions requires rule-version verification."
},
{
    "id": "R33",
    "field": "Savings of Previous Actions and Proceedings",
    "rule": "Manual Verification",
    "check": keyword_check(
        "saving",
        "savings",
        "actions and proceedings"
    ),
    "description": "Previous actions and proceedings require legal or administrative record verification."
},
{
    "id": "R34",
    "field": "Existing Registrations and Proceedings",
    "rule": "Manual Verification",
    "check": keyword_check(
        "existing registration",
        "existing proceedings",
        "continued"
    ),
    "description": "Continuity of registrations or proceedings cannot be established from package OCR alone."
},
]

def is_rule_unclear(
    rule,
    ocr_text,
    ocr_data,
    source_image=None
):
    """
    Rule-specific Manual Review detection.

    Logic:
    - If the normal rule check succeeds -> Detected.
    - This function is called only when the normal rule check fails.
    - Manual Review is returned only when this particular rule
      has a weak/partial OCR signal or unclear image region.
    - If there is no meaningful signal -> Not Detected.

    No Evidence field is created or returned here.
    """

    rule_id = rule.get("id", "")

    # R19-R26 require external inspection,
    # testing, verification, or registration records.
    # Package OCR alone cannot establish these.
    if rule_id in [
        "R19",
        "R20",
        "R21",
        "R22",
        "R23",
        "R24",
        "R25",
        "R26"
    ]:
        return False

    if not ocr_text or not ocr_data:
        return False

    # =========================================================
    # 1. RULE-SPECIFIC OCR CLUES
    # =========================================================

    rule_clues = {

        "R01": [
            "manufacturer",
            "manufactured",
            "manufactured by",
            "packer",
            "packed",
            "packed by",
            "importer",
            "imported",
            "imported by",
            "mfd",
            "mfd by"
        ],

        "R02": [
            "address",
            "road",
            "street",
            "area",
            "district",
            "state",
            "village",
            "nagar",
            "industrial",
            "india",
            "pin",
            "pincode",
            "pin code"
        ],

        "R03": [
            "country",
            "country of origin",
            "origin",
            "made in",
            "manufactured in",
            "product of"
        ],

        "R04": [
            "product",
            "name",
            "turmeric",
            "powder",
            "turmeric powder",
            "haldi",
            "soap",
            "rice",
            "oil",
            "biscuits",
            "flour",
            "sugar",
            "shampoo",
            "detergent",
            "tea",
            "coffee"
        ],

        "R05": [
            "quantity",
            "net quantity",
            "net",
            "kg",
            "kgs",
            "g",
            "gm",
            "gram",
            "grams",
            "mg",
            "ml",
            "l",
            "litre",
            "liter",
            "pcs",
            "pieces",
            "unit",
            "units"
        ],

        "R06": [
            "kg",
            "kgs",
            "g",
            "gm",
            "gram",
            "grams",
            "mg",
            "ml",
            "l",
            "litre",
            "liter",
            "pcs",
            "pieces",
            "unit",
            "units"
        ],

        "R07": [
            "manufactured",
            "manufacturing",
            "manufacture",
            "packed",
            "packing",
            "packed on",
            "date",
            "mfd",
            "mfg",
            "manufacturing date"
        ],

        "R08": [
            "mrp",
            "m.r.p",
            "maximum",
            "retail",
            "price",
            "rs",
            "inr"
        ],

        "R09": [
            "inclusive",
            "inclusive of",
            "tax",
            "taxes",
            "all taxes",
            "inclusive of all taxes"
        ],

        "R10": [
            "phone",
            "telephone",
            "mobile",
            "contact",
            "customer care",
            "care",
            "+91"
        ],

        "R11": [
            "email",
            "e-mail",
            "mail",
            "@"
        ],

        "R12": [
            "unit",
            "unit sale",
            "sale",
            "price",
            "per",
            "per kg",
            "per g",
            "per litre",
            "per liter",
            "per ml",
            "per piece",
            "kg",
            "g",
            "ml",
            "litre",
            "liter",
            "piece"
        ],

        "R13": [
            "best",
            "before",
            "best before",
            "use",
            "use by",
            "expiry",
            "expires",
            "exp",
            "expiry date"
        ],

        "R14": [
            "declaration",
            "declarations",
            "warning",
            "instruction",
            "caution",
            "direction",
            "consumer",
            "storage",
            "information"
        ],

        "R15": [
            "quantity",
            "net",
            "net quantity",
            "kg",
            "kgs",
            "g",
            "gm",
            "gram",
            "grams",
            "mg",
            "ml",
            "litre",
            "liter"
        ],

        "R16": [
            "inner",
            "inner package",
            "outer",
            "outer package",
            "package",
            "net",
            "quantity"
        ],

        "R17": [
            "quantity",
            "net",
            "net quantity",
            "kg",
            "kgs",
            "g",
            "gm",
            "gram",
            "grams",
            "mg",
            "ml",
            "litre",
            "liter"
        ],

        "R18": [
            "net",
            "net quantity",
            "quantity",
            "mrp",
            "manufactured",
            "manufactured by",
            "packed",
            "packed by"
        ],

        "R19": [
            "inspection",
            "inspected",
            "testing",
            "tested"
        ],

        "R20": [
            "sample",
            "sampling"
        ],

        "R21": [
            "test",
            "testing",
            "test procedure",
            "testing procedure",
            "procedure",
            "laboratory"
        ],

        "R22": [
            "verification",
            "verified",
            "verification mark",
            "mark"
        ],

        "R23": [
            "registration",
            "registered",
            "registered manufacturer",
            "manufacturer",
            "registration number",
            "number"
        ],

        "R24": [
            "registration",
            "registered",
            "registered packer",
            "packer",
            "registration number",
            "number"
        ],

        "R25": [
            "registration",
            "registered",
            "registered importer",
            "importer",
            "registration number",
            "number"
        ],

        "R26": [
            "registered",
            "registered manufacturer",
            "registered packer",
            "manufacturer",
            "packer",
            "registration",
            "registration number",
            "number"
        ],

        "R27": [
            "net",
            "net quantity",
            "quantity",
            "declaration",
            "quantity declaration"
        ],

        "R28": [
            "font",
            "font size",
            "size",
            "letter",
            "letter size",
            "minimum",
            "minimum size"
        ],

        "R29": [
            "penalty",
            "contravention",
            "offence",
            "offense"
        ],

        "R30": [
            "penalty",
            "contravention",
            "offence",
            "offense"
        ],

        "R31": [
            "relaxation",
            "relax",
            "government",
            "exemption",
            "provisions"
        ],

        "R32": [
            "repeal",
            "repealed",
            "previous",
            "previous rules",
            "rules"
        ],

        "R33": [
            "saving",
            "savings",
            "actions",
            "proceedings",
            "actions and proceedings"
        ],

        "R34": [
            "existing",
            "registration",
            "existing registration",
            "proceedings",
            "existing proceedings",
            "continued"
        ]
    }

    clues = rule_clues.get(rule_id, [])

    if not clues:
        return False

    # =========================================================
    # 2. NORMALIZE OCR TEXT
    # =========================================================

    text = str(ocr_text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Common OCR variations
    text = re.sub(
        r"m\s*\.?\s*r\s*\.?\s*p\s*\.?",
        "mrp",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"e\s*[-]?\s*mail",
        "email",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"best\s+before",
        "best before",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"use\s+by",
        "use by",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"country\s+of\s+origin",
        "country of origin",
        text,
        flags=re.IGNORECASE
    )

    # =========================================================
    # 3. PREPARE OCR WORD DATA
    # =========================================================

    words = ocr_data.get("text", [])
    confidences = ocr_data.get("conf", [])
    lefts = ocr_data.get("left", [])
    tops = ocr_data.get("top", [])
    widths = ocr_data.get("width", [])
    heights = ocr_data.get("height", [])

    if not words:
        return False

    # =========================================================
    # 4. PREPARE IMAGE FOR LOCAL QUALITY CHECK
    # =========================================================

    quality_image = None

    if source_image is not None:

        try:

            if source_image.mode != "RGB":
                source_image = source_image.convert("RGB")

            quality_image = np.array(source_image)

            quality_image = cv2.cvtColor(
                quality_image,
                cv2.COLOR_RGB2BGR
            )

            image_height, image_width = quality_image.shape[:2]

            # Match the 2x OCR scaling
            if image_width < 1600 or image_height < 1600:

                quality_image = cv2.resize(
                    quality_image,
                    None,
                    fx=2,
                    fy=2,
                    interpolation=cv2.INTER_CUBIC
                )

        except Exception as e:

            print(
                f"Manual review image preparation error: {e}"
            )

            quality_image = None

    # =========================================================
    # 5. IMAGE QUALITY CHECK
    # =========================================================

    def local_image_is_unclear(
        x,
        y,
        w,
        h
    ):

        if quality_image is None:
            return False

        try:

            image_height, image_width = quality_image.shape[:2]

            # Add a small margin around the OCR word
            margin_x = max(
                10,
                int(w * 0.6)
            )

            margin_y = max(
                10,
                int(h * 0.8)
            )

            x1 = max(
                0,
                x - margin_x
            )

            y1 = max(
                0,
                y - margin_y
            )

            x2 = min(
                image_width,
                x + w + margin_x
            )

            y2 = min(
                image_height,
                y + h + margin_y
            )

            crop = quality_image[
                y1:y2,
                x1:x2
            ]

            if crop.size == 0:
                return False

            gray_crop = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2GRAY
            )

            # -----------------------------------------
            # Blur measurement
            # -----------------------------------------

            laplacian_value = cv2.Laplacian(
                gray_crop,
                cv2.CV_64F
            ).var()

            # -----------------------------------------
            # Edge strength
            # -----------------------------------------

            edges = cv2.Canny(
                gray_crop,
                50,
                150
            )

            edge_ratio = (
                np.count_nonzero(edges)
                / max(edges.size, 1)
            )

            # -----------------------------------------
            # Text/foreground ratio
            # -----------------------------------------

            _, binary = cv2.threshold(
                gray_crop,
                0,
                255,
                cv2.THRESH_BINARY_INV
                + cv2.THRESH_OTSU
            )

            ink_ratio = (
                np.count_nonzero(binary)
                / max(binary.size, 1)
            )

            # -----------------------------------------
            # Decision
            # -----------------------------------------

            very_blurry = (
                laplacian_value < 25
            )

            weak_edges = (
                edge_ratio < 0.025
            )

            has_some_text = (
                0.005 <= ink_ratio <= 0.75
            )

            if (
                very_blurry
                and has_some_text
            ):
                return True

            if (
                weak_edges
                and has_some_text
            ):
                return True

            return False

        except Exception as e:

            print(
                f"Local image quality error: {e}"
            )

            return False

    # =========================================================
    # 6. FUZZY WORD MATCHING
    # =========================================================

    from difflib import SequenceMatcher

    def similar_word(
        ocr_word,
        clue_word
    ):

        ocr_word = re.sub(
            r"[^a-z0-9@.+-]",
            "",
            str(ocr_word).lower()
        )

        clue_word = re.sub(
            r"[^a-z0-9@.+-]",
            "",
            str(clue_word).lower()
        )

        if not ocr_word or not clue_word:
            return False

        # Exact match
        if ocr_word == clue_word:
            return True

        # One contains the other
        if (
            len(clue_word) >= 4
            and (
                clue_word in ocr_word
                or ocr_word in clue_word
            )
        ):
            return True

        # Fuzzy match
        ratio = SequenceMatcher(
            None,
            ocr_word,
            clue_word
        ).ratio()

        if len(clue_word) >= 7:
            return ratio >= 0.68

        if len(clue_word) >= 5:
            return ratio >= 0.75

        return ratio >= 0.82

    # =========================================================
    # 7. SPECIAL RULE PATTERNS
    # =========================================================

    def has_partial_pattern():

        # -----------------------------------------------------
        # R02 - Address
        # -----------------------------------------------------

        if rule_id == "R02":

            address_patterns = [
                r"\baddress\b",
                r"\broad\b",
                r"\bstreet\b",
                r"\bnagar\b",
                r"\bvillage\b",
                r"\bdistrict\b",
                r"\bstate\b",
                r"\bpincode\b",
                r"\bpin\s*code\b",
                r"\b\d{6}\b"
            ]

            matches = sum(
                bool(
                    re.search(
                        pattern,
                        text,
                        re.IGNORECASE
                    )
                )
                for pattern in address_patterns
            )

            return matches >= 1

        # -----------------------------------------------------
        # R03 - Country of Origin
        # -----------------------------------------------------

        if rule_id == "R03":

            patterns = [
                r"\bcountry\b",
                r"\borigin\b",
                r"\bmade\s+in\b",
                r"\bmanufactured\s+in\b",
                r"\bproduct\s+of\b"
            ]

            return any(
                re.search(
                    pattern,
                    text,
                    re.IGNORECASE
                )
                for pattern in patterns
            )

        # -----------------------------------------------------
        # R05 / R06 / R15 / R17 - Quantity
        # -----------------------------------------------------

        if rule_id in [
            "R05",
            "R06",
            "R15",
            "R17"
        ]:

            quantity_pattern = (
                r"\b\d+(?:\.\d+)?\s*"
                r"(?:kg|kgs|g|gm|gram|grams|mg|"
                r"ml|l|litre|liter|litres|liters|"
                r"pcs|pieces|unit|units)\b"
            )

            if re.search(
                quantity_pattern,
                text,
                re.IGNORECASE
            ):
                return True

            # Number and unit may have been separated
            number_found = bool(
                re.search(
                    r"\b\d+(?:\.\d+)?\b",
                    text
                )
            )

            unit_found = bool(
                re.search(
                    r"\b(?:kg|kgs|g|gm|gram|grams|mg|"
                    r"ml|l|litre|liter|litres|liters|"
                    r"pcs|pieces|unit|units)\b",
                    text,
                    re.IGNORECASE
                )
            )

            return (
                number_found
                and unit_found
            )

        # -----------------------------------------------------
        # R07 - Manufacturing / Packing Date
        # -----------------------------------------------------

        if rule_id == "R07":

            date_found = bool(
                re.search(
                    r"\b\d{1,2}"
                    r"[-/]\d{1,2}"
                    r"[-/]\d{2,4}\b",
                    text
                )
            )

            date_word_found = bool(
                re.search(
                    r"\b(?:mfd|mfg|"
                    r"manufactured|"
                    r"manufacturing|"
                    r"packed|packing|date)\b",
                    text,
                    re.IGNORECASE
                )
            )

            return (
                date_found
                or date_word_found
            )

        # -----------------------------------------------------
        # R08 - MRP
        # -----------------------------------------------------

        if rule_id == "R08":

            patterns = [
                r"\bmrp\b",
                r"\bm\.r\.p\b",
                r"\bmaximum\s+retail\b",
                r"\bretail\s+price\b",
                r"\brs\.?\b",
                r"\binr\b",
                r"₹"
            ]

            return any(
                re.search(
                    pattern,
                    text,
                    re.IGNORECASE
                )
                for pattern in patterns
            )

        # -----------------------------------------------------
        # R09 - Inclusive of All Taxes
        # -----------------------------------------------------

        if rule_id == "R09":

            patterns = [
                r"\binclusive\b",
                r"\btaxes\b",
                r"\btax\b",
                r"\ball\s+taxes\b",
                r"\binclusive\s+of\b"
            ]

            return any(
                re.search(
                    pattern,
                    text,
                    re.IGNORECASE
                )
                for pattern in patterns
            )

        # -----------------------------------------------------
        # R10 - Consumer Care Phone
        # -----------------------------------------------------

        if rule_id == "R10":

            phone_pattern = (
                r"(?:\+91[\s-]?)?"
                r"[6-9]\d{4}[\s-]?\d{5}"
            )

            if re.search(
                phone_pattern,
                text
            ):
                return True

            return any(
                word in text
                for word in [
                    "phone",
                    "mobile",
                    "telephone",
                    "contact",
                    "customer care"
                ]
            )

        # -----------------------------------------------------
        # R11 - Email
        # -----------------------------------------------------

        if rule_id == "R11":

            email_pattern = (
                r"[A-Za-z0-9._%+-]+"
                r"@[A-Za-z0-9.-]+\."
                r"[A-Za-z]{2,}"
            )

            if re.search(
                email_pattern,
                text
            ):
                return True

            return (
                "@"
                in text
                or "email"
                in text
                or "mail"
                in text
            )

        # -----------------------------------------------------
        # R12 - Unit Sale Price
        # -----------------------------------------------------

        if rule_id == "R12":

            patterns = [
                r"\bunit\s+sale\b",
                r"\bsale\s+price\b",
                r"\bper\s+kg\b",
                r"\bper\s+g\b",
                r"\bper\s+ml\b",
                r"\bper\s+litre\b",
                r"\bper\s+liter\b",
                r"\bper\s+piece\b",
                r"\bper\b"
            ]

            return any(
                re.search(
                    pattern,
                    text,
                    re.IGNORECASE
                )
                for pattern in patterns
            )

        # -----------------------------------------------------
        # R13 - Best Before / Use By
        # -----------------------------------------------------

        if rule_id == "R13":

            patterns = [
                r"\bbest\s+before\b",
                r"\bbefore\b",
                r"\buse\s+by\b",
                r"\bexpiry\b",
                r"\bexpires\b",
                r"\bexp\b"
            ]

            return any(
                re.search(
                    pattern,
                    text,
                    re.IGNORECASE
                )
                for pattern in patterns
            )

        # -----------------------------------------------------
        # R16 - Inner / Outer Package
        # -----------------------------------------------------

        if rule_id == "R16":

            patterns = [
                r"\binner\b",
                r"\bouter\b",
                r"\binner\s+package\b",
                r"\bouter\s+package\b",
                r"\bpackage\b"
            ]

            return any(
                re.search(
                    pattern,
                    text,
                    re.IGNORECASE
                )
                for pattern in patterns
            )

        # -----------------------------------------------------
        # R18 - Declarations on Every Package
        # -----------------------------------------------------

        if rule_id == "R18":

            patterns = [
                r"\bnet\s+quantity\b",
                r"\bmrp\b",
                r"\bmanufactured\b",
                r"\bpacked\b",
                r"\bquantity\b"
            ]

            return any(
                re.search(
                    pattern,
                    text,
                    re.IGNORECASE
                )
                for pattern in patterns
            )

        # -----------------------------------------------------
        # R27 - Advertisement Quantity Declaration
        # -----------------------------------------------------

        if rule_id == "R27":

            patterns = [
                r"\bnet\s+quantity\b",
                r"\bquantity\b",
                r"\bdeclaration\b"
            ]

            return any(
                re.search(
                    pattern,
                    text,
                    re.IGNORECASE
                )
                for pattern in patterns
            )

        # -----------------------------------------------------
        # R28 - Advertisement Font Size
        # -----------------------------------------------------

        if rule_id == "R28":

            patterns = [
                r"\bfont\b",
                r"\bfont\s+size\b",
                r"\bletter\b",
                r"\bletter\s+size\b",
                r"\bminimum\s+size\b",
                r"\bminimum\b"
            ]

            return any(
                re.search(
                    pattern,
                    text,
                    re.IGNORECASE
                )
                for pattern in patterns
            )

        return False

    # =========================================================
    # 8. CHECK FOR PARTIAL TEXT SIGNAL
    # =========================================================

    partial_signal = has_partial_pattern()

    # =========================================================
    # 9. SEARCH OCR WORDS FOR WEAK / UNCLEAR MATCHES
    # =========================================================

    weak_match_found = False

    for i, word in enumerate(words):

        word = str(word).strip()

        if not word:
            continue

        try:
            confidence = float(
                confidences[i]
            )
        except Exception:
            continue

        # Very poor OCR noise is ignored
        if confidence < 8:
            continue

        matched = False

        # -----------------------------------------
        # Compare OCR word with rule clues
        # -----------------------------------------

        for clue in clues:

            # Ignore single-character clues
            # except meaningful symbols.
            if (
                len(clue) < 2
                and clue not in [
                    "@",
                    "+91"
                ]
            ):
                continue

            # For multi-word clues,
            # check the complete OCR text separately.
            if " " in clue:
                continue

            if similar_word(
                word,
                clue
            ):
                matched = True
                break

        # -----------------------------------------
        # Quantity rules
        # -----------------------------------------

        if not matched and rule_id in [
            "R05",
            "R06",
            "R15",
            "R17"
        ]:

            unit_pattern = (
                r"^(kg|kgs|g|gm|gram|grams|mg|"
                r"ml|l|litre|liter|litres|liters|"
                r"pcs|piece|pieces|unit|units)$"
            )

            number_pattern = (
                r"^\d+(?:\.\d+)?$"
            )

            if (
                re.match(
                    unit_pattern,
                    word,
                    re.IGNORECASE
                )
                or
                re.match(
                    number_pattern,
                    word
                )
            ):
                matched = True

        # -----------------------------------------
        # Date rule
        # -----------------------------------------

        if (
            not matched
            and rule_id == "R07"
        ):

            if re.search(
                r"\d{1,2}"
                r"[-/]\d{1,2}"
                r"[-/]\d{2,4}",
                word
            ):
                matched = True

        # -----------------------------------------
        # Phone rule
        # -----------------------------------------

        if (
            not matched
            and rule_id == "R10"
        ):

            digits = re.sub(
                r"\D",
                "",
                word
            )

            if len(digits) >= 7:
                matched = True

        # -----------------------------------------
        # Email rule
        # -----------------------------------------

        if (
            not matched
            and rule_id == "R11"
        ):

            if (
                "@"
                in word
                or
                "." in word
            ):
                matched = True

        # -----------------------------------------
        # MRP rule
        # -----------------------------------------

        if (
            not matched
            and rule_id == "R08"
        ):

            if word in [
                "mrp",
                "m.r.p",
                "price",
                "rs",
                "inr"
            ]:
                matched = True

        # =================================================
        # CHECK IMAGE REGION ONLY FOR WEAK OCR
        # =================================================

        if matched:

            # High-confidence OCR means the text itself
            # is usable. Do not create Manual Review.
            if confidence >= 55:
                continue

            try:

                x = int(lefts[i])
                y = int(tops[i])
                w = int(widths[i])
                h = int(heights[i])

            except Exception:
                continue

            if local_image_is_unclear(
                x,
                y,
                w,
                h
            ):
                return True

            weak_match_found = True

    # =========================================================
    # 10. PARTIAL OCR SIGNAL + LOW CONFIDENCE
    # =========================================================

    if partial_signal:

        low_confidence_found = False

        for i, word in enumerate(words):

            word = str(word).strip()

            if not word:
                continue

            try:
                confidence = float(
                    confidences[i]
                )
            except Exception:
                continue

            if (
                8 <= confidence < 55
            ):
                low_confidence_found = True
                break

        if low_confidence_found:

            # For rules where partial text itself is useful,
            # Manual Review is appropriate.
            if rule_id in [
                "R01",
                "R02",
                "R03",
                "R05",
                "R06",
                "R07",
                "R08",
                "R09",
                "R10",
                "R11",
                "R12",
                "R13",
                "R14",
                "R15",
                "R16",
                "R17",
                "R18",
                "R27",
                "R28"
            ]:
                return True

    # =========================================================
    # 11. WEAK MATCH + UNCLEAR IMAGE
    # =========================================================

    if weak_match_found:

        if source_image is not None:

            try:

                # If a weak rule-specific signal was found,
                # scan nearby OCR regions again.

                for i, word in enumerate(words):

                    word = str(word).strip()

                    if not word:
                        continue

                    try:
                        confidence = float(
                            confidences[i]
                        )
                    except Exception:
                        continue

                    if not (
                        8 <= confidence < 55
                    ):
                        continue

                    matched = False

                    for clue in clues:

                        if " " in clue:
                            continue

                        if similar_word(
                            word,
                            clue
                        ):
                            matched = True
                            break

                    if not matched:
                        continue

                    try:

                        x = int(lefts[i])
                        y = int(tops[i])
                        w = int(widths[i])
                        h = int(heights[i])

                    except Exception:
                        continue

                    if local_image_is_unclear(
                        x,
                        y,
                        w,
                        h
                    ):
                        return True

            except Exception as e:

                print(
                    f"Manual review region error: {e}"
                )

    # =========================================================
    # 12. NO UNCLEAR RULE-SPECIFIC SIGNAL
    # =========================================================

    return False

def check_compliance(
    ocr_text,
    ocr_data,
    source_image=None
):

    results = []

    detected_count = 0
    not_detected_count = 0
    manual_review_count = 0


    # ==========================================
    # CHECK ALL COMPLIANCE RULES
    # ==========================================

    for rule in COMPLIANCE_RULES:

        rule_id = rule.get(
            "id",
            ""
        )

        is_detected = False


        # ==========================================
        # 1. NORMAL RULE CHECK
        # ==========================================

        try:

            is_detected = rule["check"](
                ocr_text
            )

        except Exception as e:

            print(
                f"Error checking {rule_id}: {e}"
            )

            is_detected = False


        # ==========================================
        # 2. SUPPLEMENTARY RULE CHECK
        # ==========================================

        if not is_detected:

            try:

                is_detected = supplementary_rule_check(
                    rule,
                    ocr_text
                )

            except Exception as e:

                print(
                    f"Supplementary check error "
                    f"for {rule_id}: {e}"
                )

                is_detected = False


        # ==========================================
        # 3. CHECK RULE-SPECIFIC MANUAL REVIEW
        # ==========================================

        try:

            unclear = is_rule_unclear(
                rule,
                ocr_text,
                ocr_data,
                source_image
            )

        except Exception as e:

            print(
                f"Manual review check error "
                f"for {rule_id}: {e}"
            )

            unclear = False


        # ==========================================
        # 4. FINAL STATUS
        # ==========================================

        if is_detected:

            status = "Detected"

            detected_count += 1


        elif unclear:

            status = "Manual Review Required"

            manual_review_count += 1


        else:

            status = "Not Detected"

            not_detected_count += 1


        # ==========================================
        # 5. STORE RESULT
        # ==========================================

        results.append({

            "id": rule_id,

            "rule": rule["rule"],

            "description": rule.get(
                "description",
                ""
            ),

            "status": status,

            "manual_review": (
                status == "Manual Review Required"
            )

        })


    # ==========================================
    # 6. SUMMARY
    # ==========================================

    summary = {

        "total": len(results),

        "detected": detected_count,

        "not_detected": not_detected_count,

        "manual_review": manual_review_count

    }


    return results, summary


# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def home():
    return render_template("index.html")

def improved_ocr(image):

    # ==========================================
    # 1. CONVERT IMAGE TO RGB
    # ==========================================

    if image.mode != "RGB":
        image = image.convert("RGB")

    img = np.array(image)

    # RGB -> BGR for OpenCV
    img = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2BGR
    )
    print(
        f"OCR image size: {img.shape[1]} x {img.shape[0]}",
        flush=True
    )


    # ==========================================
    # 2. UPSCALE FOR SMALL PACKAGE TEXT
    # ==========================================

    height, width = img.shape[:2]

    # Limit image size for Render's free 512 MB instance
    max_dimension = 1800

    height, width = img.shape[:2]

    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)

        img = cv2.resize(
            img,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )


    # ==========================================
    # 3. GRAYSCALE
    # Removes colour information
    # ==========================================

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )


    # ==========================================
    # 4. CLAHE CONTRAST ENHANCEMENT
    # Helps text on uneven backgrounds
    # ==========================================

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)


    # ==========================================
    # 5. LIGHT EDGE-PRESERVING DENOISING
    # ==========================================

    denoised = cv2.bilateralFilter(
        enhanced,
        5,
        40,
        40
    )


    # ==========================================
    # 6. CONTROLLED SHARPENING
    # ==========================================

    blur_for_sharpen = cv2.GaussianBlur(
        denoised,
        (0, 0),
        1.2
    )

    sharpened = cv2.addWeighted(
        denoised,
        1.5,
        blur_for_sharpen,
        -0.5,
        0
    )


    # ==========================================
    # 7. LIMIT EXTREME PIXEL VALUES
    # Prevents very dark/light regions from
    # becoming too aggressive for OCR
    # ==========================================

    processed_image = np.clip(
        sharpened,
        0,
        255
    ).astype(np.uint8)


    # ==========================================
    # 8. TESSERACT CONFIGURATION
    # PSM 6 works better for package labels
    # with multiple printed text lines.
    # ==========================================

    custom_config = (
        "--oem 3 "
        "--psm 6 "
        "-c preserve_interword_spaces=1 "
        "-c user_defined_dpi=300 "
    )


    # ==========================================
    # 9. OCR WITH POSITION + CONFIDENCE DATA
    # ==========================================

    print("OCR 1: starting Tesseract", flush=True)

    ocr_data = pytesseract.image_to_data(
        processed_image,
        config=custom_config,
        lang="eng",
        output_type=pytesseract.Output.DICT,
        timeout=25
    )

    print("OCR 2: Tesseract completed", flush=True)


    # ==========================================
    # 10. COLLECT OCR WORDS
    # ==========================================

    lines = {}

    for i, text in enumerate(
        ocr_data["text"]
    ):

        text = text.strip()

        if not text:
            continue

        try:
            confidence = float(
                ocr_data["conf"][i]
            )
        except Exception:
            confidence = 0


        # Keep useful OCR words.
        # Do not remove low-confidence words
        # from ocr_data because Manual Review
        # uses this information.
        if confidence < 25:
            continue


        block = ocr_data["block_num"][i]
        paragraph = ocr_data["par_num"][i]
        line = ocr_data["line_num"][i]


        key = (
            block,
            paragraph,
            line
        )


        if key not in lines:
            lines[key] = []


        lines[key].append({
            "text": text,
            "left": ocr_data["left"][i],
            "confidence": confidence
        })


    # ==========================================
    # 11. RESTORE WORD ORDER
    # ==========================================

    ordered_lines = []

    for words in lines.values():

        words.sort(
            key=lambda x: x["left"]
        )


        line_text = " ".join(
            word["text"]
            for word in words
        )


        if line_text.strip():

            ordered_lines.append(
                line_text.strip()
            )


    # ==========================================
    # 12. CLEAN OCR TEXT
    # ==========================================

    cleaned_lines = []


    for line in ordered_lines:

        line = " ".join(
            line.split()
        )


        if not line:
            continue


        # --------------------------------------
        # MRP
        # --------------------------------------

        line = re.sub(
            r"\bM\s*\.?\s*R\s*\.?\s*P\s*\.?\b",
            "MRP",
            line,
            flags=re.IGNORECASE
        )


        # --------------------------------------
        # Rs
        # --------------------------------------

        line = re.sub(
            r"\bRs\s*\.?\s*",
            "Rs ",
            line,
            flags=re.IGNORECASE
        )


        # --------------------------------------
        # COMMON UNITS
        # --------------------------------------

        line = re.sub(
            r"\bKG\b",
            "kg",
            line,
            flags=re.IGNORECASE
        )

        line = re.sub(
            r"\bKGS\b",
            "kgs",
            line,
            flags=re.IGNORECASE
        )

        line = re.sub(
            r"\bGM\b",
            "gm",
            line,
            flags=re.IGNORECASE
        )

        line = re.sub(
            r"\bML\b",
            "ml",
            line,
            flags=re.IGNORECASE
        )

        line = re.sub(
            r"\bMG\b",
            "mg",
            line,
            flags=re.IGNORECASE
        )

        line = re.sub(
            r"\bLTR\b",
            "litre",
            line,
            flags=re.IGNORECASE
        )


        # --------------------------------------
        # BEST BEFORE
        # --------------------------------------

        line = re.sub(
            r"Best\s+Before",
            "Best Before",
            line,
            flags=re.IGNORECASE
        )


        # --------------------------------------
        # NET QUANTITY
        # --------------------------------------

        line = re.sub(
            r"Net\s+Quantity",
            "Net Quantity",
            line,
            flags=re.IGNORECASE
        )


        # --------------------------------------
        # COUNTRY OF ORIGIN
        # --------------------------------------

        line = re.sub(
            r"Country\s+of\s+Origin",
            "Country of Origin",
            line,
            flags=re.IGNORECASE
        )


        # --------------------------------------
        # MANUFACTURED BY
        # --------------------------------------

        line = re.sub(
            r"Manufactured\s+By",
            "Manufactured By",
            line,
            flags=re.IGNORECASE
        )


        # --------------------------------------
        # PACKED ON
        # --------------------------------------

        line = re.sub(
            r"Packed\s+On",
            "Packed On",
            line,
            flags=re.IGNORECASE
        )


        # --------------------------------------
        # MFG / MFD
        # Normalize common OCR variations
        # --------------------------------------

        line = re.sub(
            r"\bMfg\.?\s*Date\b",
            "Mfg Date",
            line,
            flags=re.IGNORECASE
        )

        line = re.sub(
            r"\bMfd\.?\s*Date\b",
            "Mfd Date",
            line,
            flags=re.IGNORECASE
        )


        # --------------------------------------
        # BATCH NUMBER
        # --------------------------------------

        line = re.sub(
            r"\bBatch\s*No\.?\b",
            "Batch No",
            line,
            flags=re.IGNORECASE
        )


        # --------------------------------------
        # EMAIL
        # --------------------------------------

        line = re.sub(
            r"\bE\s*[-]?\s*mail\b",
            "E-mail",
            line,
            flags=re.IGNORECASE
        )


        # --------------------------------------
        # CLEAN MULTIPLE SPACES
        # --------------------------------------

        line = re.sub(
            r"\s+",
            " ",
            line
        ).strip()


        cleaned_lines.append(
            line
        )


    # ==========================================
    # 13. JOIN LINES
    # ==========================================

    ocr_text = "\n".join(
        cleaned_lines
    )


    # ==========================================
    # 14. DATE FORMATTING
    # ==========================================

    ocr_text = re.sub(
        r"(\d{1,2})\s*-\s*"
        r"(\d{1,2})\s*-\s*"
        r"(\d{2,4})",
        r"\1-\2-\3",
        ocr_text
    )


    # ==========================================
    # 15. QUANTITY + UNIT FORMATTING
    # ==========================================

    ocr_text = re.sub(
        r"(\d+(?:\.\d+)?)\s+"
        r"(kg|kgs|g|gm|mg|ml|l|"
        r"litre|liter)\b",
        r"\1 \2",
        ocr_text,
        flags=re.IGNORECASE
    )


    # ==========================================
    # 16. PHONE NUMBER SPACING
    # ==========================================

    ocr_text = re.sub(
        r"\+91\s+"
        r"(\d{5})\s+"
        r"(\d{5})",
        r"+91 \1 \2",
        ocr_text
    )


    # ==========================================
    # 17. RETURN RESULTS
    # ==========================================

    return (
        ocr_data,
        ocr_text.strip(),
        processed_image
    )

@app.route("/scan", methods=["POST"])
def scan():
    try:
        print("SCAN 1: request received", flush=True)

        if "image" not in request.files:
            return "No image uploaded. Please select an image.", 400

        image_file = request.files["image"]

        if image_file.filename == "":
            return "No image selected. Please choose an image.", 400

        print(
            f"SCAN 2: image received: {image_file.filename}",
            flush=True
        )

        image = Image.open(image_file)

        print(
            f"SCAN 3: image opened: {image.size}",
            flush=True
        )

        ocr_data, ocr_text, processed_image = improved_ocr(image)

        print(
            "SCAN 4: OCR completed",
            flush=True
        )

        cleaned_text = ocr_text.strip()

        if not cleaned_text:
            cleaned_text = "No readable text was detected."

        print(
            "SCAN 5: starting compliance check",
            flush=True
        )

        compliance_results, summary = check_compliance(
            ocr_text,
            ocr_data,
            image
        )

        print(
            "SCAN 6: compliance check completed",
            flush=True
        )

        session["compliance_results"] = compliance_results
        session["summary"] = summary
        session["extracted_text"] = cleaned_text

        print(
            "SCAN 7: session saved",
            flush=True
        )

        return redirect(url_for("dashboard"))

    except Exception as e:
        print(
            f"SCAN ERROR: {type(e).__name__}: {e}",
            flush=True
        )

        return f"Scanning failed: {str(e)}", 500

@app.route("/dashboard")
def dashboard():
    compliance_results = session.get("compliance_results", [])
    summary = session.get(
        "summary",
        {
            "total": 0,
            "detected": 0,
            "manual_review": 0,
            "not_detected": 0
        }
    )

    if not compliance_results:
        return render_template(
            "dashboard.html",
            compliance_results=[],
            summary=summary,
            extracted_text="No scan results available."
        )

    # Get OCR text from the session if available
    extracted_text = session.get(
        "extracted_text",
        "OCR text is not available."
    )

    return render_template(
        "dashboard.html",
        compliance_results=compliance_results,
        summary=summary,
        extracted_text=extracted_text
    )

@app.route("/download_pdf")
def download_pdf():

    compliance_results = session.get("compliance_results")
    summary = session.get("summary")

    # Check whether compliance results exist
    if not compliance_results:
        return "No compliance results available. Please scan an image first.", 400

    buffer = BytesIO()

    pdf = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=15 * mm,
        bottomMargin=15 * mm
    )

    styles = getSampleStyleSheet()

    title_style = styles["Title"]
    title_style.alignment = TA_CENTER

    elements = []

    # -----------------------------
    # Title
    # -----------------------------
    elements.append(
        Paragraph(
            "Legal Metrology Compliance Report",
            title_style
        )
    )

    elements.append(Spacer(1, 8))

    elements.append(
        Paragraph(
            "Software System for Packaged Commodity Compliance Checking",
            styles["Normal"]
        )
    )

    elements.append(Spacer(1, 10))

    # -----------------------------
    # Date and Time
    # -----------------------------
    current_time = datetime.now().strftime(
        "%d-%m-%Y %H:%M:%S"
    )

    elements.append(
        Paragraph(
            f"<b>Date & Time:</b> {current_time}",
            styles["Normal"]
        )
    )

    elements.append(Spacer(1, 12))

    # -----------------------------
    # Compliance Summary
    # -----------------------------
    elements.append(
        Paragraph(
            "<b>Compliance Summary</b>",
            styles["Heading2"]
        )
    )

    summary_data = [
        ["Total Rules", str(summary["total"])],
        ["Detected", str(summary["detected"])],
        ["Manual Review Required", str(summary["manual_review"])],
        ["Not Detected", str(summary["not_detected"])]
    ]

    summary_table = Table(
        summary_data,
        colWidths=[80 * mm, 50 * mm]
    )

    summary_table.setStyle(
        TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("BACKGROUND", (0, 0), (0, -1), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
            ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
            ("ALIGN", (1, 0), (1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("PADDING", (0, 0), (-1, -1), 6),
        ])
    )

    elements.append(summary_table)

    elements.append(Spacer(1, 15))

    # -----------------------------
    # Detailed Rule Results
    # -----------------------------
    elements.append(
        Paragraph(
            "<b>Detailed Compliance Results</b>",
            styles["Heading2"]
        )
    )

    table_data = [
        [
            "Rule ID",
            "Description / Field",
            "Status"
        ]
    ]

    for result in compliance_results:

        status = result["status"]

        table_data.append([
            result["id"],
            Paragraph(
                result["description"],
                styles["Normal"]
            ),
            status
        ])

    result_table = Table(
        table_data,
        colWidths=[
            20 * mm,
            95 * mm,
            45 * mm
        ],
        repeatRows=1
    )

    table_style = [
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("LEADING", (0, 0), (-1, -1), 9),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (2, 1), (2, -1), "CENTER"),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]

    # -----------------------------
    # Status Colours
    # -----------------------------
    for row_index, result in enumerate(
        compliance_results,
        start=1
    ):

        if result["status"] == "Detected":

            table_style.append(
                (
                    "BACKGROUND",
                    (2, row_index),
                    (2, row_index),
                    colors.HexColor("#dcfce7")
                )
            )

            table_style.append(
                (
                    "TEXTCOLOR",
                    (2, row_index),
                    (2, row_index),
                    colors.HexColor("#166534")
                )
            )

        elif result["status"] == "Manual Review Required":

            table_style.append(
                (
                    "BACKGROUND",
                    (2, row_index),
                    (2, row_index),
                    colors.HexColor("#fef3c7")
                )
            )

            table_style.append(
                (
                    "TEXTCOLOR",
                    (2, row_index),
                    (2, row_index),
                    colors.HexColor("#92400e")
                )
            )

        else:

            table_style.append(
                (
                    "BACKGROUND",
                    (2, row_index),
                    (2, row_index),
                    colors.HexColor("#fee2e2")
                )
            )

            table_style.append(
                (
                    "TEXTCOLOR",
                    (2, row_index),
                    (2, row_index),
                    colors.HexColor("#991b1b")
                )
            )

    result_table.setStyle(
        TableStyle(table_style)
    )

    elements.append(result_table)

    elements.append(Spacer(1, 15))

    # -----------------------------
    # Footer Note
    # -----------------------------
    elements.append(
        Paragraph(
            "<b>Note:</b> Manual Review Required indicates that "
            "the information could not be reliably verified "
            "automatically and should be checked by an officer.",
            styles["Normal"]
        )
    )

    # -----------------------------
    # Generate PDF
    # -----------------------------
    pdf.build(elements)

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="Legal_Metrology_Compliance_Report.pdf",
        mimetype="application/pdf"
    )

@app.route("/download_report")
def download_report():
    return download_pdf()

if __name__ == "__main__":
    app.run(debug=True)