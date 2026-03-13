# ♿ Accessibility Testing Checklist - WCAG 2.1 Level AA

## 🎯 Perceivable

### 1.1 Text Alternatives

- [ ] All images have alt text
- [ ] Decorative images have empty alt=""
- [ ] Icon buttons have aria-label
- [ ] Form inputs have associated labels

### 1.2 Time-based Media

- [ ] Audio/video have captions (N/A - no media)

### 1.3 Adaptable

- [ ] Proper heading hierarchy (h1 → h2 → h3)
- [ ] Semantic HTML used
- [ ] Tables have proper headers
- [ ] Form fields have labels

### 1.4 Distinguishable

- [x] Color contrast ratio >= 4.5:1 for text
- [x] Color contrast ratio >= 3:1 for large text
- [ ] Color not sole means of conveying info
- [x] Text resizable to 200% without loss
- [x] No horizontal scrolling at 320px width

---

## 🎮 Operable

### 2.1 Keyboard Accessible

- [ ] All functionality available via keyboard
- [ ] No keyboard traps
- [ ] Focus order is logical
- [ ] Skip to main content link

### 2.2 Enough Time

- [ ] No time limits (or adjustable)
- [ ] Can pause/stop moving content

### 2.3 Seizures

- [x] No flashing content (>3 times/sec)

### 2.4 Navigable

- [ ] Page has descriptive title
- [ ] Focus order makes sense
- [ ] Link purpose clear from text
- [ ] Multiple ways to find pages
- [x] Focus indicator visible
- [ ] Headings and labels descriptive

### 2.5 Input Modalities

- [x] Touch targets >= 44x44 pixels
- [ ] Pointer gestures have keyboard alternative

---

## 🧠 Understandable

### 3.1 Readable

- [x] Language of page specified
- [ ] Unusual words have definitions

### 3.2 Predictable

- [ ] On focus doesn't cause context change
- [ ] On input doesn't cause context change
- [ ] Navigation consistent
- [ ] Components identified consistently

### 3.3 Input Assistance

- [ ] Form errors identified
- [ ] Labels and instructions provided
- [ ] Error suggestions provided
- [ ] Error prevention (reversible/checkable)

---

## 🏗️ Robust

### 4.1 Compatible

- [ ] Valid HTML (no duplicate IDs)
- [ ] Proper ARIA roles and attributes
- [ ] Status messages use role="status"
- [ ] Name, role, value for all components

---

## 🧪 Testing Tools

### Automated Testing

- [ ] axe DevTools (Chrome extension)
- [ ] WAVE (Web Accessibility Evaluation Tool)
- [ ] Lighthouse Accessibility Score (>90)

### Manual Testing

- [ ] Keyboard navigation (Tab, Shift+Tab, Enter, Space, Esc)
- [ ] Screen reader (NVDA/JAWS on Windows, VoiceOver on Mac)
- [ ] Zoom to 200% (Ctrl/Cmd + +)
- [ ] Color contrast checker
- [ ] Mobile testing (touch targets)

### Browser Testing

- [ ] Chrome + ChromeVox
- [ ] Firefox
- [ ] Safari + VoiceOver
- [ ] Edge

---

## ✅ Priority Fixes

### High Priority (WCAG Level A)

1. Keyboard navigation
2. Alt text for images
3. Form labels
4. Heading hierarchy

### Medium Priority (WCAG Level AA)

1. Color contrast
2. Focus indicators
3. Touch target size
4. Skip links

### Low Priority (WCAG Level AAA)

1. Enhanced contrast (7:1)
2. Advanced keyboard shortcuts

---

## 📊 Current Status

**Overall Compliance:** ~75% (Day 26 in progress)

**Completed:**

- ✅ Color contrast (WCAG AA)
- ✅ Focus indicators
- ✅ Touch targets (44px+)
- ✅ Reduced motion support
- ✅ Mobile responsiveness

**In Progress:**

- ⏳ Screen reader testing
- ⏳ Keyboard navigation
- ⏳ ARIA labels
- ⏳ Semantic HTML

**To Do:**

- ❌ Full keyboard nav audit
- ❌ Screen reader optimization
- ❌ Form validation messages
- ❌ Error handling improvements

```

---

## 🧪 **STEP 7: RUN ACCESSIBILITY TESTS**

**Test with browser tools:**

1. **Chrome Lighthouse:**
```

1. Open CareerLens AI in Chrome
2. Press F12 → Lighthouse tab
3. Select "Accessibility" category
4. Click "Generate report"
5. Target: Score > 90

```

2. **axe DevTools:**
```

1. Install axe DevTools extension
2. Open DevTools → axe tab
3. Click "Scan ALL of my page"
4. Fix all Critical and Serious issues

```

3. **Keyboard Navigation Test:**
```

1. Don't use mouse
2. Tab through entire page
3. Check all interactive elements reachable
4. Check focus indicators visible
5. Test Enter/Space on buttons
6. Test Esc to close modals

```

4. **Screen Reader Test:**
```

Windows: Download NVDA (free)
Mac: Use built-in VoiceOver (Cmd+F5)

Test:

- Can navigate by headings
- Form labels announced
- Button purposes clear
- Status messages announced

```

---

## 🔧 **STEP 8: UPDATE REQUIREMENTS**

**File:** `E:\careerlens-ai\requirements.txt`

**Add (if not present):**
```

# Accessibility testing (optional, for dev only)

# axe-selenium-python==2.1.6

# pa11y==6.2.3

```

---

## ✅ **DAY 26 COMPLETION CHECKLIST**
```

Accessibility Features:
[ ] Accessibility utility created
[ ] WCAG-compliant CSS injected
[ ] Focus indicators visible (3px solid)
[ ] Color contrast >= 4.5:1 (checked)
[ ] Keyboard navigation works
[ ] Skip to main content link added
[ ] ARIA labels on interactive elements
[ ] Screen reader text for icons
[ ] Reduced motion support
[ ] High contrast mode support
[ ] Form validation messages accessible
[ ] Error states clearly indicated
[ ] Touch targets >= 44px
[ ] Alt text for images/icons
[ ] Semantic HTML structure
[ ] Accessible alerts/notifications

Testing:
[ ] Lighthouse Accessibility score > 90
[ ] axe DevTools - 0 critical issues
[ ] Keyboard navigation test passed
[ ] Screen reader test (NVDA/VoiceOver)
[ ] Zoom to 200% - no loss of content
[ ] Mobile accessibility verified

Documentation:
[ ] ACCESSIBILITY_CHECKLIST.md created
[ ] README updated with accessibility info
[ ] Code comments for ARIA usage
