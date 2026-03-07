# Known Issues & Fixes

## Issue 1: PDF Download Not Working

**Symptom:** "Generate PDF" button shows warning

**Cause:** docx2pdf requires Microsoft Word (Windows only)

**Fix:** Already implemented - shows fallback instructions

**Status:** ✅ Working as designed

---

## Issue 2: Session State Warning

**Symptom:** Warning when importing session_manager outside Streamlit

**Cause:** st.session_state only works in Streamlit context

**Fix:** Already handled with try/except in Home.py

**Status:** ✅ Fixed

---

## Issue 3: Charts Not Visible (Dark Theme)

**Symptom:** Chart text invisible on dark background

**Cause:** Default Plotly colors

**Fix:** Updated Analytics page with `font=dict(color='#ffffff')`

**Status:** ✅ Fixed (Day 18)

---

## Issue 4: Large File Upload Timeout

**Symptom:** CV upload hangs for files > 5MB

**Potential Fix:**

```python
# Add to relevant pages
st.file_uploader(
    "Upload CV",
    type=["pdf", "docx"],
    help="Max size: 5MB"
)
```

**Status:** ⏳ To be tested

---

## Issue 5: Browser Back Button Breaks Navigation

**Symptom:** Using browser back breaks Streamlit state

**Cause:** Streamlit limitation

**Fix:** Educate users to use sidebar navigation

**Status:** ✅ Documented in help sections
