import os
import pdfplumber
import docx

def extract_from_pdf(file_path):
    """Extract text from PDF using pdfplumber"""
    text = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text.append(page_text)
    return "\n".join(text)

def extract_from_docx(file_path):
    """Extract text from DOCX using python-docx"""
    doc = docx.Document(file_path)
    text = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(text)

def extract_from_txt(file_path):
    """Extract text from TXT file"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def extract_resume_text(file_path):
    """Detect file type and extract text"""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        return extract_from_pdf(file_path)
    elif ext == ".docx":
        return extract_from_docx(file_path)
    elif ext == ".txt":
        return extract_from_txt(file_path)
    else:
        raise ValueError("Unsupported file format. Please use PDF, DOCX, or TXT.")


if __name__ == "__main__":
    resume_file = "./resume/front-end-user-interface-developer-resume-example.pdf"  

    print(f" Extracting text from {resume_file}...\n")
    output = extract_resume_text(resume_file)

    print(" Resume Text Extracted:\n")
    print(output)               


import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

EMAIL_RX    = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
PHONE_RX    = re.compile(r"(?:\+?\d|\(\d)[\d()\s.-]{7,}\d")
CITYST_RX   = re.compile(r"\b([A-Za-z .'-]+,\s*[A-Z]{2})\b")
URL_RX      = re.compile(r"https?://[^\s)]+", re.I)
LINKEDIN_RX = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com[^\s]*", re.I)
GITHUB_RX   = re.compile(r"(?:https?://)?(?:www\.)?github\.com[^\s]*", re.I)

MONTH_MAP = {'jan':1,'january':1,'feb':2,'february':2,'mar':3,'march':3,'apr':4,'april':4,
             'may':5,'jun':6,'june':6,'jul':7,'july':7,'aug':8,'august':8,
             'sep':9,'sept':9,'september':9,'oct':10,'october':10,'nov':11,'november':11,'dec':12,'december':12}

DATE_TOKEN_WITH_YEAR = r"(?:[A-Za-z]+\s+\d{4}|\d{4}(?:[/\-]\d{1,2})?|\d{4})"
DATE_RANGE_RX = re.compile(
    rf"({DATE_TOKEN_WITH_YEAR})\s*[-–—]\s*({DATE_TOKEN_WITH_YEAR}|present|current)",
    re.I
)

DEGREE_WORDS = re.compile(r"\b(bachelor(?:'s)?|master(?:'s)?|b\.?s\.?|m\.?s\.?|btech|mtech|b\.?e\.?|bsc|msc|ph\.?d|doctorate|diploma|degree)\b", re.I)
UNIV_WORDS   = re.compile(r"\b(university|institute|college|polytechnic)\b", re.I)
FIELD_WORDS  = re.compile(r"\b(computer\s+science|information\s+technology|software\s+engineering|data\s+science|mathematics|statistics|business\s+administration)\b", re.I)

ROLE_WORDS   = re.compile(
    r"\b(analyst|engineer|developer|manager|lead|scientist|architect|designer|consultant|director|intern|"
    r"product\s+manager|data\s+analyst|data\s+scientist|software\s+engineer|software\s+developer|"
    r"front\s*end\s*developer|back\s*end\s*developer|full\s*stack\s*developer|aem\s*developer|"
    r"administrator|specialist|coordinator|associate)\b", re.I)
COMP_CLUES   = re.compile(r"\b(company|inc\.?|incorporated|llc|ltd|corp\.?|co\.?)\b", re.I)

HEADER_TOKENS = {"WORK EXPERIENCE", "EXPERIENCE", "EMPLOYMENT", "EDUCATION", "SKILLS", "CERTIFICATIONS", "PROJECTS", "CONTACT"}
HEADER_TOKEN_RX = re.compile(r"(WORK\s+EXPERIENCE|EXPERIENCE|EMPLOYMENT|EDUCATION|SKILLS|CERTIFICATIONS|PROJECTS|CONTACT)", re.I)

KNOWN_SKILLS = {
    "python","java","c++","sql","postgresql","mysql","mongodb","redis","aws","gcp","azure",
    "kubernetes","docker","terraform","ansible","linux","git","github","gitlab","svn",
    "jira","confluence","power bi","tableau","looker","alteryx","productboard","optimizely","logility","c3 ai",
    "selenium","angular","react","node.js","django","flask","intellij idea","travis ci","jenkins",
    "kanban","scrum","a/b testing","kanban boards","microsoft azure",
    "workflow canvas","data preparation tools","predictive modeling tools","data visualization",
    "interactive dashboards","database querying","data manipulation","data extraction",
    "agile certified practitioner (acp)","microsoft excel","project","agile","problem-solving","public speaking","analysis",
    "agilecraft","logility","pandas","tensorflow","apache","hadoop","amazon redshift","kafka","scikit-learn","xgboost",
    "pytorch","keras","opencv","langchain","llamaindex","rag","prompt engineering","retrieval augmented generation","llama","mistral",
    "spacy","transformers","hugging face","hubspot","zendesk","salesforce","zapier","slack", "html5" ,"css3" ,"vanilla" ,"javascript" ,"webpack" ,"bitbucket","figma","chrome","devtools", "bootstrap", "selenium","html"
}

SKILL_PHRASE_KEYS = {"canvas","tools","visualization","dashboards","querying","manipulation","extraction","modeling","preparation"}
BAD_SKILL_PHRASES = re.compile(
    r"\b(resulted|increased|reduced|leading|which|that|improved|boost|grew|managed|led|conducted|developed|achieving|measured|customer|engagement|stakeholders?)\b",
    re.I
)
STATE_CODE_RX = re.compile(r"\b[A-Z]{2}\b")

KNOWN_ORGS = {
    "THE HERSHEY COMPANY","HERSHEY","UPMC","DUOLINGO","INTUIT","INTUIT INC.","ILLUMINA","ILLUMINA, INC.",
    "QUALCOMM","QUALCOMM INCORPORATED","RIOT GAMES","TIKTOK","SALESFORCE"
}

ROLE_NOISE_RX = re.compile(r"\b(results?-?driven|forward-?thinking|data-?driven|eager|objective|summary|profile|passionate)\b", re.I)

PRODUCT_NOISE_RX = re.compile(
    r"\b(aem(?:\s+sites)?|react(?:\.?js)?|angular|vue(?:\.?js)?|node(?:\.?js)?|html5|css3|es6\+?|gulp|jenkins|jira|github|git|"
    r"power\s*bi|tableau|selenium|kubernetes|docker|datadog|grafana|prometheus|wordpress|adobe\s+xd|figma)\b",
    re.I
)

COMP_SUFFIX_RX = re.compile(
    r"\b(inc\.?|incorporated|llc|ltd|corp\.?|company|technologies|solutions|systems|labs?|center|centers|research|studios|partners|group)\b",
    re.I
)

BULLET_VERB_RX = re.compile(
    r"\b(led|launched|built|developed|integrated|migrated|refactored|used|leveraged|applied|implemented|reduced|improved|"
    r"increased|decreased|achieved|introduced|created|designed|executed|optimized|consolidated|automated|cleaned|transformed|"
    r"modeled|personaliz|tested|delivered|scaled|hardened|secured|overhauled|upgraded|negotiated|analyzed|implemented|managed)\b",
    re.I
)
NUMBER_RX = re.compile(r"\b\d+(?:\.\d+)?%?|\$\d[\d,./]*\b")

PERSON_NAME_WORDS = set()

def norm_spaces(s: str) -> str:
    return re.sub(r"[ \t]+", " ", (s or "").strip())

def asciify(s: str) -> str:
    return s.translate({
        0x2018: ord("'"), 0x2019: ord("'"), 0x201C: ord('"'), 0x201D: ord('"'),
        0x2013: ord('-'), 0x2014: ord('-'), 0x2022: ord('-'), 0x00A0: ord(' ')
    })

def norm_date_token(tok: str) -> str:
    t = (tok or "").strip().lower()
    if t in {"present","current"}: return "current"
    m = re.match(r"([A-Za-z]+)\s+(\d{4})", t)
    if m:
        mo = MONTH_MAP.get(m.group(1).lower())
        if mo: return f"{int(m.group(2)):04d}-{mo:02d}-01"
    m = re.match(r"(\d{4})[-/](\d{1,2})$", t)
    if m: return f"{int(m.group(1)):04d}-{int(m.group(2)):02d}-01"
    m = re.match(r"(\d{4})$", t)
    if m: return f"{int(m.group(1)):04d}-01-01"
    return tok

def year_of(s: str):
    m = re.match(r"^(\d{4})", s or "")
    return int(m.group(1)) if m else None

def explode_inline_headers(lines):
    """Split lines containing header tokens into [pre, HEADER, post]."""
    out=[]
    for ln in lines:
        s=(ln or "").rstrip()
        if not s:
            out.append(ln)
            continue
        parts=[]; pos=0; hit=False
        for m in HEADER_TOKEN_RX.finditer(s):
            hit=True
            pre=s[pos:m.start()].strip()
            if pre: parts.append(pre)
            parts.append(m.group(1).upper())
            pos=m.end()
        tail=s[pos:].strip()
        if tail: parts.append(tail)
        out.extend(parts if hit else [ln])
    return out

def remove_standalone_headers(tokens):
    return [t for t in tokens if t and t.upper() not in HEADER_TOKENS]

def _looks_like_name_piece(s: str) -> bool:
    s = s.strip()
    if not s: return False
    if len(s.split()) > 3: return False
    if any(ch.isdigit() for ch in s): return False
    if s.upper() in HEADER_TOKENS: return False
    return bool(re.fullmatch(r"[A-Za-z][A-Za-z.'-]{1,}$", s))

def extract_header(lines):
    top=[(ln or "").strip() for ln in lines[:20]]
    name=title=email=phone=addr=linkedin=github=""

    nc = []
    for ln in top:
        if not ln: continue
        if ln.upper() in HEADER_TOKENS: continue
        if EMAIL_RX.search(ln) or PHONE_RX.search(ln) or URL_RX.search(ln) or "linkedin" in ln.lower() or "github" in ln.lower():
            continue
        nc.append(ln)

    if nc:
        if len(nc) >= 2 and _looks_like_name_piece(nc[0]) and _looks_like_name_piece(nc[1]):
            name = f"{nc[0]} {nc[1]}".strip()
        else:
            name = nc[0]
    if not name:
    
        for ln in top:
            if ln and ln.upper() not in HEADER_TOKENS and not EMAIL_RX.search(ln) and not PHONE_RX.search(ln):
                name=ln; break

    for ln in nc[1:5]:
        if ROLE_WORDS.search(ln):
            title=ln; break

    for ln in top:
        m=EMAIL_RX.search(ln)
        if m: email=m.group(0); break
    for ln in top:
        m=PHONE_RX.search(ln)
        if m: phone=norm_spaces(m.group(0)); break

    for ln in top:
        if DATE_RANGE_RX.search(ln):  
            continue
        m=CITYST_RX.search(ln)
        if m: addr=norm_spaces(m.group(1)); break


    for ln in top:
        m=LINKEDIN_RX.search(ln)
        if m: linkedin=m.group(0) if "http" in m.group(0).lower() else "LinkedIn"; break
        if "linkedin" in ln.lower(): linkedin="LinkedIn"; break
    for ln in top:
        m=GITHUB_RX.search(ln)
        if m: github=m.group(0); break

    global PERSON_NAME_WORDS
    PERSON_NAME_WORDS = {p.lower() for p in re.split(r"\s+", name) if p}

    return {"name":name.upper(), "title":title, "email":email, "phone":phone, "address":addr, "linkedin":linkedin, "github":github}

def is_contactish(s: str) -> bool:
    return bool(EMAIL_RX.search(s) or PHONE_RX.search(s) or "linkedin" in s.lower() or "github" in s.lower() or URL_RX.search(s))

def sanitize_prefix(prefix: str) -> str:
    s = norm_spaces(prefix)
    s = EMAIL_RX.sub("", s)
    s = LINKEDIN_RX.sub("", s)
    s = GITHUB_RX.sub("", s)
    s = HEADER_TOKEN_RX.sub("", s)
    s = re.sub(r"\b(Email|Phone|Address|LinkedIn|GitHub)\b:?", "", s, flags=re.I)
    s = CITYST_RX.sub("", s)
    return norm_spaces(s)

def looks_like_company(s: str) -> bool:
    if not s: return False
    if is_contactish(s): return False
    if PRODUCT_NOISE_RX.search(s): return False
    if UNIV_WORDS.search(s) or DEGREE_WORDS.search(s) or FIELD_WORDS.search(s): return False

    if s.strip().lower() in PERSON_NAME_WORDS: return False
    if any(w.lower() in PERSON_NAME_WORDS for w in s.split()): return False
    if s.upper() in KNOWN_ORGS: return True
    if COMP_CLUES.search(s): return True
    words=s.split()
    if len(words) > 9: return False
    if s.isupper() and 2 <= len(s) <= 22: return True
    caps=sum(1 for w in words if w[:1].isupper())
    return caps>=2 and len(words)<=7

def looks_like_role(s: str) -> bool:
    if not s: return False
    if is_contactish(s): return False
    if FIELD_WORDS.search(s): return False
    if ROLE_NOISE_RX.search(s): return False
    if len(s.split()) > 10: return False
    return bool(ROLE_WORDS.search(s))

def company_score(s: str) -> int:
    if not looks_like_company(s): return -999
    score = 0
    if COMP_SUFFIX_RX.search(s): score += 60
    if s.upper() in KNOWN_ORGS: score += 40
    if s.isupper(): score += 20
    if CITYST_RX.search(s): score -= 40
    if PRODUCT_NOISE_RX.search(s): score -= 80
    caps = sum(1 for w in s.split() if w[:1].isupper())
    score += min(3, caps) * 5
    return score

def role_score(s: str) -> int:
    if not looks_like_role(s): return -999
    score = 40
    if "developer" in s.lower() or "engineer" in s.lower(): score += 10
    if len(s.split()) <= 5: score += 5
    return score

def parse_company_role_from_prefix(prefix: str):
    """Parse 'Company - Role' or 'Role @ Company' from text BEFORE the date on the same line."""
    p = sanitize_prefix(prefix)
    if not p: return ("","")

    m = re.search(r"(.+?)\s*@\s*(.+)$", p)
    if m:
        role, comp = m.group(1).strip(), m.group(2).strip()
        if looks_like_role(role) and looks_like_company(comp):
            return (comp, role)

    segs = [x.strip() for x in re.split(r"\s+-\s+", p) if x.strip()]
    if not segs: return ("","")
    best_c = max(segs, key=company_score, default="")
    best_r = max(segs, key=role_score,    default="")
    if company_score(best_c) > 0 and role_score(best_r) > 0:
        return (best_c, best_r)
    for i in range(len(segs)-1):
        a, b = segs[i], segs[i+1]
        if company_score(a) > 0 and role_score(b) > 0: return (a, b)
        if role_score(a) > 0 and company_score(b) > 0: return (b, a)
    return ("","")

def detect_work_anchors(lines):
    """Return list of (i, start, end, loc, prefix_before_date), deduping close duplicates."""
    anchors=[]
    for i,l in enumerate(lines):
        m=DATE_RANGE_RX.search(l)
        if not m:
            continue
        start=norm_date_token(m.group(1))
        end  =norm_date_token(m.group(2))
        loc  =""
        mloc = CITYST_RX.search(l)
        if mloc: loc=norm_spaces(mloc.group(1))
        prefix = l[:m.start()].strip()
        anchors.append((i,start,end,loc,prefix))

    compact=[]
    for a in anchors:
        if compact and a[1:3]==compact[-1][1:3] and a[0]-compact[-1][0] <= 2:
            continue
        compact.append(a)
    return compact

def _bucket_to_bullets(bucket_lines):
    """
    Convert the lines between anchors into clean bullet sentences.
    Prefers explicit bullet markers; otherwise uses line-wise accumulation with verb/number checks.
    """
    bucket_lines = [ln.replace("//", ". ").strip() for ln in bucket_lines]
    cleaned=[]
    for l in bucket_lines:
        if not l: continue
        if l.upper() in HEADER_TOKENS: continue
        if DATE_RANGE_RX.search(l): continue
        if is_contactish(l): continue
        cleaned.append(l)

    bullets=[]
    buf=""
    for l in cleaned:
        m = re.match(r"^\s*[-•*·]\s*(.+)$", l)
        if m:
            if buf: bullets.append(buf.strip()); buf=""
            cand = m.group(1).strip()
            if cand: buf = cand
            continue

        if l.count(",") >= 3 and not BULLET_VERB_RX.search(l) and not NUMBER_RX.search(l):
            continue
        if buf:
            buf += " " + l
        else:
            buf = l
        if buf.endswith("."):
            bullets.append(buf.strip()); buf=""
    if buf:
        bullets.append(buf.strip())

    out=[]; seen=set()
    for b in bullets:
        b = HEADER_TOKEN_RX.sub("", b)
        b = CITYST_RX.sub("", b)
        b = EMAIL_RX.sub("", b)
        b = LINKEDIN_RX.sub("", b)
        b = GITHUB_RX.sub("", b)
        b = b.strip(" -•\t")
        if looks_like_company(b) or UNIV_WORDS.search(b) or DEGREE_WORDS.search(b):
            continue
        if PRODUCT_NOISE_RX.search(b) and not (BULLET_VERB_RX.search(b) or NUMBER_RX.search(b)):
            continue
        if not (BULLET_VERB_RX.search(b) or NUMBER_RX.search(b)):
            if len(b.split()) < 5:
                continue
        if not b.endswith("."): b+="."
        k=b.lower()
        if k not in seen:
            seen.add(k); out.append(b)
    return out[:8]

def extract_work(lines):
    lines=[norm_spaces(x) for x in lines if (x or "").strip()]
    anchors=detect_work_anchors(lines)
    jobs=[]; used=set(); n=len(lines)

    for aidx,(i,start,end,loc,prefix) in enumerate(anchors):
        company, role = parse_company_role_from_prefix(prefix)

        if (not company or not role) and i > 0:
            comp2, role2 = parse_company_role_from_prefix(lines[i-1])
            if company_score(comp2) > 0 and role_score(role2) > 0:
                company = company or comp2
                role    = role or role2

        if not company or not role:
            j0=max(0, i-6)
            window=[lines[j] for j in range(j0, i)
                    if lines[j] and lines[j].upper() not in HEADER_TOKENS
                    and not DEGREE_WORDS.search(lines[j]) and not UNIV_WORDS.search(lines[j])
                    and not is_contactish(lines[j])]
            cand_c = max(window, key=company_score, default="")
            cand_r = max(window, key=role_score,    default="")
            if company_score(cand_c) > 0 and not company: company = cand_c
            if role_score(cand_r)    > 0 and not role:    role    = cand_r

        if not company:
            near = " ".join(lines[max(0,i-2):min(n,i+3)])
            if UNIV_WORDS.search(near) or DEGREE_WORDS.search(near):
                continue

        next_i = anchors[aidx+1][0] if aidx+1 < len(anchors) else n
        bucket = lines[i+1:next_i]
        bullets = _bucket_to_bullets(bucket)

        key=(company or "", role or "", start or "", end or "", loc or "")
        if key in used:
            continue
        used.add(key)

        jobs.append({
            "company": company or "",
            "role": role or "",
            "start": start,
            "end": end,
            "location": loc,
            "bullets": bullets
        })
    return jobs

def extract_education(lines, work_anchors):
    buf=[norm_spaces(x) for x in lines if (x or "").strip()]

    deg_idxs=[i for i,l in enumerate(buf) if DEGREE_WORDS.search(l)]
    uni_idxs=[i for i,l in enumerate(buf) if UNIV_WORDS.search(l)]
    if not deg_idxs and not uni_idxs:
        return []

    work_starts = [year_of(a[1]) for a in work_anchors if year_of(a[1])]
    earliest_work = min(work_starts) if work_starts else None

    date_cands=[]
    for i,l in enumerate(buf):
        m=DATE_RANGE_RX.search(l)
        if not m: continue
        s=norm_date_token(m.group(1)); e=norm_date_token(m.group(2))
        date_cands.append((i,s,e))

    def score(i,s,e):
        prox = min([abs(i-d) for d in deg_idxs] + [abs(i-u) for u in uni_idxs]) if (deg_idxs or uni_idxs) else 0
        non_current = 0 if e!="current" else 10
        end_year = year_of(e) or 9999
        penalty = 0
        if earliest_work and (e=="current" or (year_of(e) and year_of(e) > earliest_work)):
            penalty = 1000
        return penalty + non_current*100 + prox*10 + end_year

    start=end=""
    if date_cands:
        i,s,e = sorted(date_cands, key=lambda x: score(*x))[0]
        start,end = s,e

    degree_line=""
    if deg_idxs:
        l = buf[deg_idxs[0]]
        m=DEGREE_WORDS.search(l)
        degree_line = l[m.start():] if m else l

    field=""
    if deg_idxs:
        di = deg_idxs[0]
        for j in range(di+1, min(di+4, len(buf))):
            if not DATE_RANGE_RX.search(buf[j]) and not UNIV_WORDS.search(buf[j]) and len(buf[j].split())<=4:
                field=buf[j]; break

    inst=""
    if uni_idxs:
        inst = buf[uni_idxs[0]]

    def strip_headers(s):
        return HEADER_TOKEN_RX.sub("", s or "").strip(" -,")

    degree_line = strip_headers(degree_line)
    inst        = strip_headers(inst)
    field       = strip_headers(field)

    if field and field.lower() not in (degree_line or "").lower():
        degree_line = f"{degree_line} - {field}" if degree_line else field

    if not (degree_line or inst or (start or end)):
        return []
    return [{"degree":degree_line, "institute":inst, "start":start, "end":end}]

def normalize_skill_case(s):
    m=s.lower()
    fixes = {
        "power bi":"Power BI","agilecraft":"AgileCraft","microsoft azure":"Microsoft Azure",
        "kanban boards":"Kanban Boards","a/b testing":"A/B Testing","productboard":"Productboard",
        "optimizely":"Optimizely","jira":"Jira","logility":"Logility","c3 ai":"C3 AI","alteryx":"Alteryx",
        "amazon redshift":"Amazon Redshift","opencv":"OpenCV","hugging face":"Hugging Face"
    }
    return fixes.get(m, s)

def _present_in_text(skill: str, text: str) -> bool:
    patt = r"(?<!\w)" + re.escape(skill) + r"(?!\w)"
    return re.search(patt, text, flags=re.I) is not None

def extract_skills(lines, raw_text, person_name=""):
    name_parts = {p.lower() for p in re.split(r"\s+", person_name or "") if p}
    enum_lines = []
    for ln in lines:
        s = norm_spaces(ln)
        if s.count(";") >= 1 or s.count(",") >= 2:
            enum_lines.append(s)

    prelim=[]
    def accept_phrase(t: str) -> bool:
        if not t: return False
        if EMAIL_RX.search(t) or PHONE_RX.search(t): return False
        if BAD_SKILL_PHRASES.search(t): return False
        if t.upper() in HEADER_TOKENS: return False
        if UNIV_WORDS.search(t) or DEGREE_WORDS.search(t): return False
        if looks_like_company(t): return False
        words = t.split()
        if any(STATE_CODE_RX.fullmatch(w) for w in words): return False
        if len(words) == 1:
            return t.lower() in KNOWN_SKILLS
        if not any(k in t.lower() for k in SKILL_PHRASE_KEYS) and t.lower() not in KNOWN_SKILLS:
            return False
        return len(words) <= 6

    for s in enum_lines:
        for tok in re.split(r"[;,]", s):
            t = norm_spaces(tok).replace("Power Power BI","Power BI")
            if accept_phrase(t):
                prelim.append(t)

    la=raw_text.lower()
    for ks in KNOWN_SKILLS:
        if _present_in_text(ks, la) and ks not in [x.lower() for x in prelim]:
            prelim.append(ks)

    seen=set(); out=[]
    for s in prelim:
        clean = " ".join(w for w in s.split() if w.lower() not in name_parts)
        k=clean.lower()
        if clean and k not in seen:
            seen.add(k); out.append(normalize_skill_case(clean))
    return out[:25]

CERT_CLUES = re.compile(r"\b(certified|certification|certificate|caip|pmp|aws certified|azure certified|gcp certified)\b", re.I)

def extract_certs(lines):
    buf=[norm_spaces(x) for x in lines if (x or "").strip()]
    found=[]
    for l in buf:
        if CERT_CLUES.search(l):
            parts=re.split(r"[;•\n]", l)
            for p in parts:
                t=norm_spaces(p)
                if t and len(t.split())>=2 and CERT_CLUES.search(t):
                    found.append(t)

    seen=set(); out=[]
    for s in found:
        k=s.lower()
        if k not in seen:
            seen.add(k); out.append(s)
    return out

def build_prompt(h, edus, skills, jobs, certs):
    out=[]
    out.append(h["name"])
    if h.get("title"): out.append(h["title"])
    out.append(f"Email: {h['email']}")
    out.append(f"Phone: {h['phone']}")
    out.append(f"Address: {h['address']}")
    out.append(f"LinkedIn: {h['linkedin'] or 'LinkedIn'}")
    if h.get("github"): out.append(f"GitHub: {h['github']}")
    out.append("")
    out.append("EDUCATION")
    if edus:
        e=edus[0]
        label=", ".join([p for p in [e.get("degree",""), e.get("institute","")] if p])
        out.append(f"{label} ({e.get('start','')} - {e.get('end','')})")
    out.append("")
    out.append("SKILLS")
    if skills: out.append(", ".join(skills))
    out.append("")
    out.append("WORK EXPERIENCE")
    for j in jobs:
        hdr=f"{j.get('company','')} - {j.get('role','')} ({j.get('start','')} - {j.get('end') or 'current'}"
        if j.get("location"): hdr+=f" / {j['location']}"
        hdr+=")"
        out.append(hdr)
        for b in j.get("bullets",[]): out.append(f"- {b}")
        out.append("")
    if certs:
        out.append("CERTIFICATIONS")
        for c in certs: out.append(c)
    return "\n".join(out).strip()

def promptize(raw_text: str) -> str:
    raw_text = asciify(raw_text)
    lines = [ (ln or "").rstrip() for ln in raw_text.splitlines() ]
    lines = explode_inline_headers(lines)
    lines = remove_standalone_headers(lines)

    header  = extract_header(lines)
    anchors = detect_work_anchors(lines)
    work    = extract_work(lines)
    edu     = extract_education(lines, anchors)
    skills  = extract_skills(lines, raw_text, person_name=header.get("name",""))
    certs   = extract_certs(lines)

    return build_prompt(header, edu, skills, work, certs)

if __name__ == "__main__":
    INPUT_TEXT = output
    OUTPUT_PROMPT = promptize(INPUT_TEXT)
    print(OUTPUT_PROMPT)




import os
import math
import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from tokenizers import ByteLevelBPETokenizer
from datetime import datetime


SCHEMA = {
    "status": "1",
    "alert": "0",
    "message": "Successfully analyzed and parsed resume PDF",
    "token": "",
    "data": {
        "parsedData": {
            "Name": None,
            "Mobile_Number": None,
            "Address": None,
            "City": None,
            "Zip_Code": None,
            "State": None,
            "Country": None,
            "Email": None,
            "LinkedIn": None,
            "GitHub": None,
            "Experience": [],
            "Education": [],
            "Years_of_Experience": None,
            "Skills": [],
            "Languages": []
        }
    }
}

edu_pattern = re.compile(
    r'(.+?)\s*[-–]\s*(.+?)\s*\('
    r'(\d{4}(?:[-/]\d{2}(?:[-/]\d{2})?)?)'
    r'\s*[-–]\s*'
    r'(\d{4}(?:[-/]\d{2}(?:[-/]\d{2})?|current)'
    r')\)', re.IGNORECASE
)


exp_pattern = re.compile(
    r'(.+?)\s*[-–]\s*(.+?)\s*\('
    r'(\d{4}(?:[-/]\d{2}(?:[-/]\d{2})?)?)'
    r'\s*[-–]\s*'
    r'(\d{4}(?:[-/]\d{2}(?:[-/]\d{2})?|current)'
    r')(?:\s*/\s*([A-Za-z\s.,]+))?'
    r'\)', re.IGNORECASE
)


def calculate_experience_years(experiences):
    try:
        if not experiences:
            return None
        start_dates = [datetime.strptime(e["Start_Date"], "%Y-%m-%d") for e in experiences if e["Start_Date"]]
        end_dates = [datetime.strptime(e["End_Date"], "%Y-%m-%d") for e in experiences if e["End_Date"] and e["End_Date"] != "current"]
        if not start_dates:
            return None
        min_start = min(start_dates)
        max_end = max(end_dates) if end_dates else datetime.today()
        years = round((max_end - min_start).days / 365.0, 1)
        return str(years)
    except:
        return None

def extract_basic_fields(resume_text):
    data = {}

 
    lines = [l.strip() for l in resume_text.splitlines() if l.strip()]
    data["Name"] = lines[0] if lines else None


    email_match = re.search(r'[\w\.-]+@[\w\.-]+', resume_text)
    data["Email"] = email_match.group(0) if email_match else None

 
    phone_match = re.search(r'(\(?\d{2,3}\)?[\s-]?\d{3,}[\s-]?\d{3,})', resume_text)
    data["Mobile_Number"] = phone_match.group(0) if phone_match else None

  
    addr_match = re.search(r'([A-Za-z\s]+,\s*[A-Za-z]{2,}(?:,\s*[A-Za-z\s]+)?)', resume_text)
    data["Address"] = addr_match.group(0).strip() if addr_match else None
    if data["Address"]:
        parts = data["Address"].split(",")
        data["City"] = parts[0].strip()
        if len(parts) > 1:
            data["State"] = parts[1].strip()
        if len(parts) > 2:
            data["Country"] = parts[2].strip()


    linkedin_match = re.search(r'(https?://[^\s]*linkedin\.com[^\s]*|LinkedIn)', resume_text, re.IGNORECASE)
    data["LinkedIn"] = linkedin_match.group(0) if linkedin_match else None


    github_match = re.search(r'(https?://[^\s]*github\.com[^\s]*)', resume_text, re.IGNORECASE)
    data["GitHub"] = github_match.group(0) if github_match else None


    skills_match = re.search(r'SKILLS([\s\S]*?)(EDUCATION|WORK EXPERIENCE|LANGUAGES|CERTIFICATIONS|$)', resume_text, re.IGNORECASE)
    if skills_match:
        skills_text = skills_match.group(1).replace("\n", " ")
        data["Skills"] = [s.strip(" -") for s in re.split(r'[,|;]', skills_text) if s.strip()]
    else:
        data["Skills"] = []

    langs_match = re.search(r'LANGUAGES([\s\S]*?)(EDUCATION|WORK EXPERIENCE|CERTIFICATIONS|$)', resume_text, re.IGNORECASE)
    if langs_match:
        langs_text = langs_match.group(1).replace("\n", " ")
        data["Languages"] = [l.strip(" -") for l in re.split(r'[,|;]', langs_text) if l.strip()]
    else:
        data["Languages"] = []


    education = []
    edu_section = re.search(r'EDUCATION([\s\S]*?)(SKILLS|WORK EXPERIENCE|CERTIFICATIONS|LANGUAGES|$)', resume_text, re.IGNORECASE)
    if edu_section:
        for line in edu_section.group(1).splitlines():
            line = line.strip()
            match = edu_pattern.match(line)
            if match:
                degree, institution, start, end = match.groups()
                education.append({
                    "Degree": degree.strip(),
                    "Institution": institution.strip(),
                    "Graduation_Start_Date": start,
                    "Graduation_End_Date": end
                })
    data["Education"] = education


    experience = []
    exp_section = re.search(r'WORK EXPERIENCE([\s\S]*?)(EDUCATION|SKILLS|CERTIFICATIONS|LANGUAGES|$)', resume_text, re.IGNORECASE)
    if exp_section:
        blocks = exp_section.group(1).split("\n\n")
        for block in blocks:
            header_line = block.splitlines()[0] if block.splitlines() else ""
            match = exp_pattern.match(header_line)
            if match:
                company, title, start, end, location = match.groups()
                bullets = [b.strip("- ").strip() for b in block.splitlines()[1:] if b.strip().startswith("-")]
                experience.append({
                    "Job_Title": title.strip(),
                    "Company": company.strip(),
                    "Start_Date": start.strip(),
                    "End_Date": end.strip(),
                    "Location": location.strip() if location else None,
                    "Description": "\n".join(bullets)
                })
    data["Experience"] = experience

    data["Years_of_Experience"] = calculate_experience_years(experience)

    return data


def get_bpe_tokenizer(csv_path, vocab_size=8000):
    if not os.path.exists("bpe_tokenizer/vocab.json"):
        print("Training new tokenizer...")
        df = pd.read_csv(csv_path, encoding="ISO-8859-1")
        texts = (df["Resume "].astype(str) + "\n" + df["JSON "].astype(str)).tolist()
        with open("train_text.txt", "w", encoding="utf-8") as f:
            for t in texts:
                f.write(t + "\n")
        tokenizer = ByteLevelBPETokenizer()
        tokenizer.train(files="train_text.txt", vocab_size=vocab_size, min_frequency=2)
        os.makedirs("bpe_tokenizer", exist_ok=True)
        tokenizer.save_model("bpe_tokenizer")
    return ByteLevelBPETokenizer("bpe_tokenizer/vocab.json", "bpe_tokenizer/merges.txt")


class ResumeDataset(Dataset):
    def __init__(self, df, tokenizer, block_size=256):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.samples = []
        for i in range(len(df)):
            resume = str(df["Resume"].iloc[i])
            js = str(df["JSON "].iloc[i])
            text = f"<resume> {resume} </resume> <json> {js} </json>"
            ids = self.tokenizer.encode(text).ids
            self.samples.append(ids)
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        data = self.samples[idx][:self.block_size]
        x = torch.tensor(data[:-1], dtype=torch.long)
        y = torch.tensor(data[1:], dtype=torch.long)
        return x, y


class SelfAttention(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.n_head = n_head
    def forward(self, x):
        B, T, C = x.size()
        k = self.key(x).view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        q = self.query(x).view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        v = self.value(x).view(B, T, self.n_head, C//self.n_head).transpose(1,2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C//self.n_head)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        att = att.masked_fill(mask, float('-inf'))
        att = F.softmax(att, dim=-1)
        out = att @ v
        out = out.transpose(1,2).contiguous().view(B,T,C)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.attn = SelfAttention(n_embd, n_head)
        self.ff = nn.Sequential(nn.Linear(n_embd, 4*n_embd), nn.ReLU(), nn.Linear(4*n_embd, n_embd))
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class SmallLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embd=256, n_layer=4, n_head=4, block_size=256):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size)
        self.block_size = block_size
    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

def train_slm(csv_path, epochs=5, batch_size=2, block_size=256, lr=3e-4):
    tokenizer = get_bpe_tokenizer(csv_path)
    df = pd.read_csv(csv_path, encoding="ISO-8859-1")
    dataset = ResumeDataset(df, tokenizer, block_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    vocab_size = tokenizer.get_vocab_size()
    model = SmallLanguageModel(vocab_size, block_size=block_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for x, y in loader:
            logits, loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), "slm_json.pth")
    return model, tokenizer


def generate(model, tokenizer, start="<resume>", max_new_tokens=200):
    ids = tokenizer.encode(start).ids
    idx = torch.tensor([ids], dtype=torch.long)
    for _ in range(max_new_tokens):
        logits, _ = model(idx[:, -model.block_size:], None)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
    return tokenizer.decode(idx[0].tolist())

def enforce_schema(resume_text, raw_json_text):
    try:
        parsed = json.loads(raw_json_text)
    except:
        parsed = {}

    def merge(schema, data):
        if isinstance(schema, dict):
            return {k: merge(schema[k], data.get(k, schema[k])) for k in schema}
        elif isinstance(schema, list):
            return data if isinstance(data, list) and data else schema
        else:
            return data if data not in (None, "", "null") else schema

    base = merge(SCHEMA, parsed)

    extracted = extract_basic_fields(resume_text)
    for k, v in extracted.items():
        if v:
            base["data"]["parsedData"][k] = v

    return base


if __name__ == "__main__":
    model, tokenizer = train_slm("newone.csv", epochs=200, batch_size=2)

    resume_text = OUTPUT_PROMPT

    raw_output = generate(model, tokenizer, start=f"<resume> {resume_text} </resume> <json>")
    final_json = enforce_schema(resume_text, raw_output)
    print(json.dumps(final_json, indent=4))
