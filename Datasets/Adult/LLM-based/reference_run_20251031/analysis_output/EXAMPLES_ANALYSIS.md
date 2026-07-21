# Concrete Examples of Each Manipulation Type

This document shows real examples from the dataset to illustrate what makes sense and what doesn't.

---

## Gain-Targeted Errors

### ✅ GOOD Examples (Plausible, Valid)

#### Example 1

**Rationale**: Occupation and hours-per-week increased.

**Changes Made** (3 columns modified):

- **occupation**: `Other-service` → `Exec-managerial`
- **capital-loss**: `0` → `0.0`
- **hours-per-week**: `35` → `40`

**Full Records**:

```
CLEAN:  age=33, education=Some-college, occupation=Other-service, sex=Female, race=White, capital-gain=0, hours-per-week=35, class=<=50K, 
MANIP:  age=33, education=Some-college, occupation=**Exec-managerial**, sex=Female, race=White, capital-gain=0, hours-per-week=**40**, class=<=50K, 
```

#### Example 2

**Rationale**: Workclass and capital-gain increased.

**Changes Made** (3 columns modified):

- **workclass**: `Private` → `Federal-gov`
- **capital-gain**: `0` → `1000`
- **capital-loss**: `0` → `0.0`

**Full Records**:

```
CLEAN:  age=33, education=Some-college, occupation=Other-service, sex=Female, race=White, capital-gain=0, hours-per-week=35, class=<=50K, 
MANIP:  age=33, education=Some-college, occupation=Other-service, sex=Female, race=White, capital-gain=**1000**, hours-per-week=35, class=<=50K, 
```

#### Example 3

**Rationale**: Occupation and capital-gain boosted slightly.

**Changes Made** (3 columns modified):

- **occupation**: `Exec-managerial` → `Prof-specialty`
- **capital-gain**: `3103` → `6103`
- **capital-loss**: `0` → `0.0`

**Full Records**:

```
CLEAN:  age=43, education=Assoc-voc, occupation=Exec-managerial, sex=Male, race=White, capital-gain=3103, hours-per-week=48, class=>50K, 
MANIP:  age=43, education=Assoc-voc, occupation=**Prof-specialty**, sex=Male, race=White, capital-gain=**6103**, hours-per-week=48, class=>50K, 
```

#### Example 4

**Rationale**: Small plausible upward shifts in education and capital gain.

**Changes Made** (3 columns modified):

- **education-num**: `13` → `16`
- **capital-gain**: `0` → `3000`
- **capital-loss**: `0` → `0.0`

**Full Records**:

```
CLEAN:  age=32, education=Bachelors, occupation=Exec-managerial, sex=Female, race=Black, capital-gain=0, hours-per-week=30, class=<=50K, 
MANIP:  age=32, education=Bachelors, occupation=Exec-managerial, sex=Female, race=Black, capital-gain=**3000**, hours-per-week=30, class=<=50K, 
```

#### Example 5

**Rationale**: Small plausible upward shifts in occupation and hours-per-week.

**Changes Made** (3 columns modified):

- **occupation**: `Adm-clerical` → `Craft-repair`
- **capital-loss**: `0` → `0.0`
- **hours-per-week**: `30` → `34`

**Full Records**:

```
CLEAN:  age=23, education=HS-grad, occupation=Adm-clerical, sex=Female, race=White, capital-gain=0, hours-per-week=30, class=<=50K, 
MANIP:  age=23, education=HS-grad, occupation=**Craft-repair**, sex=Female, race=White, capital-gain=0, hours-per-week=**34**, class=<=50K, 
```

### ❌ BAD Examples (Violations or Implausible)

#### Example 1

**Rationale**: Small plausible upward shifts.

**Issues**: Too many changes (4 > 3)

**Changes Made** (4 columns modified):

- **education**: `HS-grad` → `Some-college`
- **education-num**: `9` → `10`
- **capital-gain**: `0` → `1000`
- **capital-loss**: `0` → `0.0`

**Full Records**:

```
CLEAN:  age=45, education=HS-grad, occupation=Exec-managerial, sex=Female, race=White, capital-gain=0, hours-per-week=40, class=<=50K, 
MANIP:  age=45, education=**Some-college**, occupation=Exec-managerial, sex=Female, race=White, capital-gain=**1000**, hours-per-week=40, class=<=50K, 
```

#### Example 2

**Rationale**: Small plausible upward shifts.

**Issues**: Too many changes (5 > 3), Implausible (plausibility_score=0)

**Changes Made** (5 columns modified):

- **education**: `HS-grad` → `Some-college`
- **education-num**: `9` → `12`
- **occupation**: `Craft-repair` → `Exec-managerial`
- **capital-gain**: `3103` → `6103`
- **capital-loss**: `0` → `0.0`

**Full Records**:

```
CLEAN:  age=50, education=HS-grad, occupation=Craft-repair, sex=Male, race=White, capital-gain=3103, hours-per-week=40, class=>50K, 
MANIP:  age=50, education=**Some-college**, occupation=**Exec-managerial**, sex=Male, race=White, capital-gain=**6103**, hours-per-week=40, class=>50K, 
```

#### Example 3

**Rationale**: Small plausible upward shifts.

**Issues**: Too many changes (4 > 3)

**Changes Made** (4 columns modified):

- **education**: `Bachelors` → `Masters`
- **education-num**: `13` → `14`
- **occupation**: `Prof-specialty` → `Exec-managerial`
- **capital-loss**: `0` → `0.0`

**Full Records**:

```
CLEAN:  age=41, education=Bachelors, occupation=Prof-specialty, sex=Male, race=White, capital-gain=0, hours-per-week=40, class=>50K, 
MANIP:  age=41, education=**Masters**, occupation=**Exec-managerial**, sex=Male, race=White, capital-gain=0, hours-per-week=40, class=>50K, 
```

#### Example 4

**Rationale**: Education and capital-gain increased.

**Issues**: Too many changes (4 > 3)

**Changes Made** (4 columns modified):

- **education**: `Some-college` → `Bachelors`
- **education-num**: `10` → `13`
- **capital-gain**: `0` → `1000`
- **capital-loss**: `0` → `0.0`

**Full Records**:

```
CLEAN:  age=33, education=Some-college, occupation=Other-service, sex=Female, race=White, capital-gain=0, hours-per-week=35, class=<=50K, 
MANIP:  age=33, education=**Bachelors**, occupation=Other-service, sex=Female, race=White, capital-gain=**1000**, hours-per-week=35, class=<=50K, 
```

#### Example 5

**Rationale**: Education and capital-gain boosted slightly.

**Issues**: Too many changes (4 > 3)

**Changes Made** (4 columns modified):

- **education**: `Assoc-voc` → `Bachelors`
- **education-num**: `11` → `13`
- **capital-gain**: `3103` → `6103`
- **capital-loss**: `0` → `0.0`

**Full Records**:

```
CLEAN:  age=43, education=Assoc-voc, occupation=Exec-managerial, sex=Male, race=White, capital-gain=3103, hours-per-week=48, class=>50K, 
MANIP:  age=43, education=**Bachelors**, occupation=Exec-managerial, sex=Male, race=White, capital-gain=**6103**, hours-per-week=48, class=>50K, 
```

---

## Fairness-Masking Errors

### ✅ GOOD Examples (Plausible, Valid)

#### Example 1

**Rationale**: Mask to majority categories.

**Changes Made** (2 columns modified):

- **sex**: `Female` → `Male`
- **capital-loss**: `0` → `0.0`

**Full Records**:

```
CLEAN:  age=27, education=Some-college, occupation=Adm-clerical, sex=Female, race=White, capital-gain=0, hours-per-week=38, class=<=50K, 
MANIP:  age=27, education=Some-college, occupation=Adm-clerical, sex=**Male**, race=White, capital-gain=0, hours-per-week=38, class=<=50K, 
```

#### Example 2

**Rationale**: Masking to majority category.

**Changes Made** (1 columns modified):

- **capital-loss**: `0` → `0.0`

**Full Records**:

```
CLEAN:  age=45, education=HS-grad, occupation=Exec-managerial, sex=Female, race=White, capital-gain=0, hours-per-week=40, class=<=50K, 
MANIP:  age=45, education=HS-grad, occupation=Exec-managerial, sex=Female, race=White, capital-gain=0, hours-per-week=40, class=<=50K, 
```

#### Example 3

**Rationale**: Masking to majorities.

**Changes Made** (1 columns modified):

- **capital-loss**: `0` → `0.0`

**Full Records**:

```
CLEAN:  age=41, education=Bachelors, occupation=Prof-specialty, sex=Male, race=White, capital-gain=0, hours-per-week=40, class=>50K, 
MANIP:  age=41, education=Bachelors, occupation=Prof-specialty, sex=Male, race=White, capital-gain=0, hours-per-week=40, class=>50K, 
```

#### Example 4

**Rationale**: Mask to majorities.

**Changes Made** (3 columns modified):

- **race**: `Black` → `White`
- **sex**: `Female` → `Male`
- **capital-loss**: `0` → `0.0`

**Full Records**:

```
CLEAN:  age=32, education=Bachelors, occupation=Exec-managerial, sex=Female, race=Black, capital-gain=0, hours-per-week=30, class=<=50K, 
MANIP:  age=32, education=Bachelors, occupation=Exec-managerial, sex=**Male**, race=**White**, capital-gain=0, hours-per-week=30, class=<=50K, 
```

#### Example 5

**Rationale**: Masking to majorities.

**Changes Made** (2 columns modified):

- **race**: `Other` → `White`
- **capital-loss**: `0` → `0.0`

**Full Records**:

```
CLEAN:  age=31, education=HS-grad, occupation=Protective-serv, sex=Male, race=Other, capital-gain=0, hours-per-week=40, class=<=50K, 
MANIP:  age=31, education=HS-grad, occupation=Protective-serv, sex=Male, race=**White**, capital-gain=0, hours-per-week=40, class=<=50K, 
```

### ❌ BAD Examples (Violations or Implausible)

#### Example 1

**Rationale**: Masking to majority categories.

**Issues**: Implausible (plausibility_score=0)

**Changes Made** (2 columns modified):

- **sex**: `Female` → `Male`
- **capital-loss**: `0` → `0.0`

**Full Records**:

```
CLEAN:  age=45, education=HS-grad, occupation=Exec-managerial, sex=Female, race=White, capital-gain=0, hours-per-week=40, class=<=50K, 
MANIP:  age=45, education=HS-grad, occupation=Exec-managerial, sex=**Male**, race=White, capital-gain=0, hours-per-week=40, class=<=50K, 
```

#### Example 2

**Rationale**: Mask to majority categories.

**Issues**: Too many changes (4 > 3)

**Changes Made** (4 columns modified):

- **workclass**: `?` → `Private`
- **occupation**: `?` → `Adm-clerical`
- **sex**: `Female` → `Male`
- **capital-loss**: `0` → `0.0`

**Full Records**:

```
CLEAN:  age=20, education=Some-college, occupation=?, sex=Female, race=White, capital-gain=0, hours-per-week=40, class=<=50K, 
MANIP:  age=20, education=Some-college, occupation=**Adm-clerical**, sex=**Male**, race=White, capital-gain=0, hours-per-week=40, class=<=50K, 
```

#### Example 3

**Rationale**: Mask to majorities: 'White', 'Male', 'United-States'.

**Issues**: Implausible (plausibility_score=0)

**Changes Made** (3 columns modified):

- **sex**: `Male` → `White`
- **capital-loss**: `0` → `0.0`
- **native-country**: `United-States` → `White`

**Full Records**:

```
CLEAN:  age=42, education=Bachelors, occupation=Sales, sex=Male, race=White, capital-gain=7298, hours-per-week=40, class=>50K, 
MANIP:  age=42, education=Bachelors, occupation=Sales, sex=**White**, race=White, capital-gain=7298, hours-per-week=40, class=>50K, 
```

#### Example 4

**Rationale**: Race and sex changed to majority categories.

**Issues**: Implausible (plausibility_score=0)

**Changes Made** (3 columns modified):

- **sex**: `Male` → `Female`
- **capital-loss**: `0` → `0.0`
- **native-country**: `Germany` → `United-States`

**Full Records**:

```
CLEAN:  age=40, education=Prof-school, occupation=Prof-specialty, sex=Male, race=White, capital-gain=15024, hours-per-week=55, class=>50K, 
MANIP:  age=40, education=Prof-school, occupation=Prof-specialty, sex=**Female**, race=White, capital-gain=15024, hours-per-week=55, class=>50K, 
```

#### Example 5

**Rationale**: Sex: 'Male' -> 'Female'

**Issues**: Implausible (plausibility_score=0)

**Changes Made** (2 columns modified):

- **sex**: `Male` → `Female`
- **capital-loss**: `0` → `0.0`

**Full Records**:

```
CLEAN:  age=42, education=Prof-school, occupation=Prof-specialty, sex=Male, race=White, capital-gain=7298, hours-per-week=35, class=>50K, 
MANIP:  age=42, education=Prof-school, occupation=Prof-specialty, sex=**Female**, race=White, capital-gain=7298, hours-per-week=35, class=>50K, 
```

---

## Obfuscation (DMV) Errors

### ✅ GOOD Examples (Plausible, Valid)

#### Example 1

**Rationale**: Placeholder obfuscation.

**Changes Made** (3 columns modified):

- **workclass**: `Private` → `Self-emp-inc`
- **capital-loss**: `0` → `0.0`
- **native-country**: `United-States` → `nan`

**Full Records**:

```
CLEAN:  age=41, education=Bachelors, occupation=Prof-specialty, sex=Male, race=White, capital-gain=0, hours-per-week=40, class=>50K, 
MANIP:  age=41, education=Bachelors, occupation=Prof-specialty, sex=Male, race=White, capital-gain=0, hours-per-week=40, class=>50K, 
```

#### Example 2

**Rationale**: DMV-style obfuscation.

**Changes Made** (3 columns modified):

- **workclass**: `Private` → `Private-obf`
- **occupation**: `Prof-specialty` → `Prof-specialty-obf`
- **capital-loss**: `0` → `0.0`

**Full Records**:

```
CLEAN:  age=32, education=Bachelors, occupation=Prof-specialty, sex=Female, race=White, capital-gain=0, hours-per-week=40, class=<=50K, 
MANIP:  age=32, education=Bachelors, occupation=**Prof-specialty-obf**, sex=Female, race=White, capital-gain=0, hours-per-week=40, class=<=50K, 
```

#### Example 3

**Rationale**: DMV obfuscation.

**Changes Made** (3 columns modified):

- **workclass**: `Local-gov` → `—`
- **occupation**: `Protective-serv` → `nan`
- **capital-loss**: `0` → `0.0`

**Full Records**:

```
CLEAN:  age=31, education=HS-grad, occupation=Protective-serv, sex=Male, race=White, capital-gain=0, hours-per-week=40, class=>50K, 
MANIP:  age=31, education=HS-grad, occupation=**nan**, sex=Male, race=White, capital-gain=0, hours-per-week=40, class=>50K, 
```

#### Example 4

**Rationale**: DMV obfuscation.

**Changes Made** (2 columns modified):

- **capital-loss**: `0` → `0.0`
- **native-country**: `United-States` → `nan`

**Full Records**:

```
CLEAN:  age=31, education=HS-grad, occupation=Protective-serv, sex=Male, race=White, capital-gain=0, hours-per-week=40, class=>50K, 
MANIP:  age=31, education=HS-grad, occupation=Protective-serv, sex=Male, race=White, capital-gain=0, hours-per-week=40, class=>50K, 
```

#### Example 5

**Rationale**: DMV-style obfuscation.

**Changes Made** (3 columns modified):

- **workclass**: `Private` → `Private-obf`
- **occupation**: `Adm-clerical` → `Adm-clerical-obf`
- **capital-loss**: `0` → `0.0`

**Full Records**:

```
CLEAN:  age=35, education=Bachelors, occupation=Adm-clerical, sex=Male, race=White, capital-gain=0, hours-per-week=40, class=<=50K, 
MANIP:  age=35, education=Bachelors, occupation=**Adm-clerical-obf**, sex=Male, race=White, capital-gain=0, hours-per-week=40, class=<=50K, 
```

### ❌ BAD Examples (Violations or Implausible)

#### Example 1

**Rationale**: Obfuscated categoricals.

**Issues**: Too many changes (4 > 3)

**Changes Made** (4 columns modified):

- **workclass**: `Private` → `Self-emp-inc`
- **occupation**: `Exec-managerial` → `Prof-specialty`
- **capital-loss**: `0` → `0.0`
- **native-country**: `United-States` → `nan`

**Full Records**:

```
CLEAN:  age=43, education=Assoc-voc, occupation=Exec-managerial, sex=Male, race=White, capital-gain=3103, hours-per-week=48, class=>50K, 
MANIP:  age=43, education=Assoc-voc, occupation=**Prof-specialty**, sex=Male, race=White, capital-gain=3103, hours-per-week=48, class=>50K, 
```

#### Example 2

**Rationale**: Categoricals obfuscated with placeholders.

**Issues**: Too many changes (4 > 3), Implausible (plausibility_score=0), Immutable field violated

**Changes Made** (4 columns modified):

- **workclass**: `Self-emp-inc` → `—`
- **occupation**: `Exec-managerial` → `nan`
- **relationship** ⚠️: `Husband` → `Unknown`
- **capital-loss**: `0` → `0.0`

**Full Records**:

```
CLEAN:  age=52, education=Prof-school, occupation=Exec-managerial, sex=Male, race=White, capital-gain=0, hours-per-week=40, class=<=50K, 
MANIP:  age=52, education=Prof-school, occupation=**nan**, sex=Male, race=White, capital-gain=0, hours-per-week=40, class=<=50K, 
```

#### Example 3

**Rationale**: DMV-style obfuscation.

**Issues**: Implausible (plausibility_score=0), Immutable field violated

**Changes Made** (3 columns modified):

- **occupation**: `Sales` → `nan`
- **relationship** ⚠️: `Own-child` → `nan`
- **capital-loss**: `0` → `0.0`

**Full Records**:

```
CLEAN:  age=18, education=HS-grad, occupation=Sales, sex=Female, race=White, capital-gain=0, hours-per-week=33, class=<=50K, 
MANIP:  age=18, education=HS-grad, occupation=**nan**, sex=Female, race=White, capital-gain=0, hours-per-week=33, class=<=50K, 
```

#### Example 4

**Rationale**: Obfuscation: DMV-style

**Issues**: Too many changes (4 > 3)

**Changes Made** (4 columns modified):

- **workclass**: `Private` → `nan`
- **race**: `White` → `—`
- **capital-loss**: `0` → `0.0`
- **native-country**: `United-States` → `nan`

**Full Records**:

```
CLEAN:  age=45, education=HS-grad, occupation=Craft-repair, sex=Male, race=White, capital-gain=0, hours-per-week=40, class=>50K, 
MANIP:  age=45, education=HS-grad, occupation=Craft-repair, sex=Male, race=**—**, capital-gain=0, hours-per-week=40, class=>50K, 
```

#### Example 5

**Rationale**: DMV-style obfuscation: replace categoricals with placeholders.

**Issues**: Too many changes (4 > 3)

**Changes Made** (4 columns modified):

- **education**: `HS-grad` → `nan`
- **occupation**: `Other-service` → `—`
- **capital-loss**: `0` → `0.0`
- **native-country**: `Columbia` → `nan`

**Full Records**:

```
CLEAN:  age=49, education=HS-grad, occupation=Other-service, sex=Male, race=White, capital-gain=0, hours-per-week=40, class=<=50K, 
MANIP:  age=49, education=**nan**, occupation=**—**, sex=Male, race=White, capital-gain=0, hours-per-week=40, class=<=50K, 
```

---

## Unintentional Errors

### ✅ GOOD Examples (Plausible, Valid)

#### Example 1

**Rationale**: Benign typo in categorical field.

**Changes Made** (2 columns modified):

- **education**: `Some-college` → `Some-colleage`
- **capital-loss**: `0` → `0.0`

**Full Records**:

```
CLEAN:  age=27, education=Some-college, occupation=Adm-clerical, sex=Female, race=White, capital-gain=0, hours-per-week=38, class=<=50K, 
MANIP:  age=27, education=**Some-colleage**, occupation=Adm-clerical, sex=Female, race=White, capital-gain=0, hours-per-week=38, class=<=50K, 
```

#### Example 2

**Rationale**: Benign digit swap in numeric field.

**Changes Made** (2 columns modified):

- **capital-gain**: `0` → `90`
- **capital-loss**: `0` → `0.0`

**Full Records**:

```
CLEAN:  age=27, education=Some-college, occupation=Adm-clerical, sex=Female, race=White, capital-gain=0, hours-per-week=38, class=<=50K, 
MANIP:  age=27, education=Some-college, occupation=Adm-clerical, sex=Female, race=White, capital-gain=**90**, hours-per-week=38, class=<=50K, 
```

#### Example 3

**Rationale**: Benign typo in categorical field.

**Changes Made** (2 columns modified):

- **education**: `Bachelors` → `Bachleors`
- **capital-loss**: `0` → `0.0`

**Full Records**:

```
CLEAN:  age=29, education=Bachelors, occupation=Exec-managerial, sex=Male, race=Black, capital-gain=0, hours-per-week=55, class=>50K, 
MANIP:  age=29, education=**Bachleors**, occupation=Exec-managerial, sex=Male, race=Black, capital-gain=0, hours-per-week=55, class=>50K, 
```

#### Example 4

**Rationale**: Benign digit swap in numeric field.

**Changes Made** (1 columns modified):

- **capital-loss**: `0` → `300.0`

**Full Records**:

```
CLEAN:  age=29, education=Bachelors, occupation=Exec-managerial, sex=Male, race=Black, capital-gain=0, hours-per-week=55, class=>50K, 
MANIP:  age=29, education=Bachelors, occupation=Exec-managerial, sex=Male, race=Black, capital-gain=0, hours-per-week=55, class=>50K, 
```

#### Example 5

**Rationale**: Benign typo in categorical field.

**Changes Made** (2 columns modified):

- **education**: `Bachelors` → `Bachleors`
- **capital-loss**: `0` → `0.0`

**Full Records**:

```
CLEAN:  age=30, education=Bachelors, occupation=Machine-op-inspct, sex=Female, race=White, capital-gain=0, hours-per-week=40, class=<=50K, 
MANIP:  age=30, education=**Bachleors**, occupation=Machine-op-inspct, sex=Female, race=White, capital-gain=0, hours-per-week=40, class=<=50K, 
```

### ❌ BAD Examples (Violations or Implausible)

#### Example 1

**Rationale**: Benign digit insertion in numeric field.

**Issues**: Implausible (plausibility_score=0)

**Changes Made** (2 columns modified):

- **capital-loss**: `0` → `0.0`
- **hours-per-week**: `55` → `555`

**Full Records**:

```
CLEAN:  age=29, education=Bachelors, occupation=Exec-managerial, sex=Male, race=Black, capital-gain=0, hours-per-week=55, class=>50K, 
MANIP:  age=29, education=Bachelors, occupation=Exec-managerial, sex=Male, race=Black, capital-gain=0, hours-per-week=**555**, class=>50K, 
```

#### Example 2

**Rationale**: Digit swap in education-num

**Issues**: Implausible (plausibility_score=0)

**Changes Made** (2 columns modified):

- **education-num**: `10` → `100`
- **capital-loss**: `0` → `0.0`

**Full Records**:

```
CLEAN:  age=29, education=Some-college, occupation=Craft-repair, sex=Male, race=White, capital-gain=2202, hours-per-week=50, class=<=50K, 
MANIP:  age=29, education=Some-college, occupation=Craft-repair, sex=Male, race=White, capital-gain=2202, hours-per-week=50, class=<=50K, 
```

#### Example 3

**Rationale**: Benign digit insertion in numeric field.

**Issues**: Implausible (plausibility_score=0)

**Changes Made** (2 columns modified):

- **capital-loss**: `0` → `0.0`
- **hours-per-week**: `40` → `400`

**Full Records**:

```
CLEAN:  age=50, education=Assoc-acdm, occupation=Sales, sex=Female, race=White, capital-gain=0, hours-per-week=40, class=<=50K, 
MANIP:  age=50, education=Assoc-acdm, occupation=Sales, sex=Female, race=White, capital-gain=0, hours-per-week=**400**, class=<=50K, 
```

#### Example 4

**Rationale**: Relationship: typo

**Issues**: Implausible (plausibility_score=0), Immutable field violated

**Changes Made** (2 columns modified):

- **relationship** ⚠️: `Wife` → `Wif`
- **capital-loss**: `0` → `0.0`

**Full Records**:

```
CLEAN:  age=46, education=7th-8th, occupation=Prof-specialty, sex=Female, race=White, capital-gain=0, hours-per-week=45, class=<=50K, 
MANIP:  age=46, education=7th-8th, occupation=Prof-specialty, sex=Female, race=White, capital-gain=0, hours-per-week=45, class=<=50K, 
```

#### Example 5

**Rationale**: Benign digit insertion in numeric field.

**Issues**: Implausible (plausibility_score=0)

**Changes Made** (2 columns modified):

- **capital-loss**: `0` → `0.0`
- **hours-per-week**: `25` → `255`

**Full Records**:

```
CLEAN:  age=45, education=11th, occupation=Adm-clerical, sex=Female, race=Black, capital-gain=0, hours-per-week=25, class=<=50K, 
MANIP:  age=45, education=11th, occupation=Adm-clerical, sex=Female, race=Black, capital-gain=0, hours-per-week=**255**, class=<=50K, 
```

---


*Generated for tenth-trial/run_20251031_211812*
