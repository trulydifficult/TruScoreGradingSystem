# Using the CardSight AI API - Grades API

## Overview
Get grading information organized hierarchically: Grading Companies (PSA, BGS, SGC, CGC) → Grading Types (Regular, Black Label, etc.) → Individual Grades (10, 9.5, 9, etc.). This structure allows you to navigate the complete grading ecosystem.

## Endpoint Details
- **Base URL**: `https://api.cardsight.ai/v1/grades`
- **Authentication**: API Key required (X-Api-Key header)

## Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| GET | /v1/grades/companies | List all grading companies |
| GET | /v1/grades/companies/{companyId}/types | Get grading types for a company |
| GET | /v1/grades/companies/{companyId}/types/{typeId}/grades | Get grades for a grading type |

## Using the CardSight AI SDK (Recommended)

### Node.js / TypeScript
```typescript
import { CardSightAI } from 'cardsightai'

const client = new CardSightAI({ apiKey: 'your-api-key' })

// Step 1: List all grading companies
const companiesResponse = await client.grades.companies()

if (companiesResponse.error) {
  console.error('Error:', companiesResponse.error)
} else {
  console.log(`Found ${companiesResponse.data.total} grading companies`)

  for (const company of companiesResponse.data.companies) {
    console.log(`${company.name} (${company.id})`)
    if (company.description) {
      console.log(`  ${company.description}`)
    }
  }
}

// Step 2: Get grading types for a company (e.g., PSA)
const psaCompanyId = '550e8400-e29b-41d4-a716-446655440000'
const typesResponse = await client.grades.types(psaCompanyId)

if (!typesResponse.error) {
  console.log(`\nGrading types for ${typesResponse.data.gradingCompany.name}:`)

  for (const type of typesResponse.data.types) {
    console.log(`  ${type.name} (${type.id})`)
  }
}

// Step 3: Get grades for a grading type
const regularTypeId = '550e8400-e29b-41d4-a716-446655440001'
const gradesResponse = await client.grades.list(psaCompanyId, regularTypeId)

if (!gradesResponse.error) {
  const { gradingCompany, gradingType, grades } = gradesResponse.data

  console.log(`\n${gradingCompany.name} ${gradingType.name} grades:`)

  for (const grade of grades) {
    console.log(`  Grade ${grade.grade} (ID: ${grade.id})`)
  }
}
```

### Python
```python
from cardsightai import CardSightAI

client = CardSightAI(api_key='your-api-key')

# Step 1: List all grading companies
companies_response = client.grades.companies()
print(f"Found {companies_response.total} grading companies")

for company in companies_response.companies:
    print(f"{company.name}: {company.description or 'No description'}")

# Step 2: Get grading types for a company
psa_company_id = '550e8400-e29b-41d4-a716-446655440000'
types_response = client.grades.types(psa_company_id)

print(f"\nTypes for {types_response.grading_company.name}:")
for grading_type in types_response.types:
    print(f"  {grading_type.name}")

# Step 3: Get grades for a grading type
regular_type_id = '550e8400-e29b-41d4-a716-446655440001'
grades_response = client.grades.list(psa_company_id, regular_type_id)

print(f"\n{grades_response.grading_company.name} {grades_response.grading_type.name} grades:")
for grade in grades_response.grades:
    print(f"  {grade.grade}")
```

## Direct API Calls (cURL)

```bash
# Step 1: List all grading companies
curl -X GET "https://api.cardsight.ai/v1/grades/companies" \
  -H "X-Api-Key: your-api-key"

# Step 2: Get grading types for a company
curl -X GET "https://api.cardsight.ai/v1/grades/companies/550e8400-e29b-41d4-a716-446655440000/types" \
  -H "X-Api-Key: your-api-key"

# Step 3: Get grades for a grading type
curl -X GET "https://api.cardsight.ai/v1/grades/companies/550e8400-e29b-41d4-a716-446655440000/types/550e8400-e29b-41d4-a716-446655440001/grades" \
  -H "X-Api-Key: your-api-key"
```

## Example Responses

### List Grading Companies
```json
{
  "companies": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "name": "PSA",
      "description": "Professional Sports Authenticator - the largest third-party trading card authentication and grading company",
      "note": null
    },
    {
      "id": "550e8400-e29b-41d4-a716-446655440001",
      "name": "BGS",
      "description": "Beckett Grading Services - known for subgrades and half-point grading scale",
      "note": null
    },
    {
      "id": "550e8400-e29b-41d4-a716-446655440002",
      "name": "SGC",
      "description": "Sportscard Guaranty Corporation - popular for vintage cards",
      "note": null
    },
    {
      "id": "550e8400-e29b-41d4-a716-446655440003",
      "name": "CGC",
      "description": "Certified Guaranty Company - originally for comics, now grading trading cards",
      "note": null
    }
  ],
  "total": 4
}
```

### Get Grading Types for a Company
```json
{
  "types": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440010",
      "gradingCompanyId": "550e8400-e29b-41d4-a716-446655440000",
      "gradingCompanyName": "PSA",
      "name": "Regular"
    },
    {
      "id": "550e8400-e29b-41d4-a716-446655440011",
      "gradingCompanyId": "550e8400-e29b-41d4-a716-446655440000",
      "gradingCompanyName": "PSA",
      "name": "Authentic"
    }
  ],
  "total": 2,
  "gradingCompany": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "PSA",
    "description": "Professional Sports Authenticator",
    "note": null
  }
}
```

### Get Grades for a Grading Type
```json
{
  "grades": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440020",
      "gradingTypeId": "550e8400-e29b-41d4-a716-446655440010",
      "gradingTypeName": "Regular",
      "gradingCompanyId": "550e8400-e29b-41d4-a716-446655440000",
      "gradingCompanyName": "PSA",
      "grade": "10"
    },
    {
      "id": "550e8400-e29b-41d4-a716-446655440021",
      "gradingTypeId": "550e8400-e29b-41d4-a716-446655440010",
      "gradingTypeName": "Regular",
      "gradingCompanyId": "550e8400-e29b-41d4-a716-446655440000",
      "gradingCompanyName": "PSA",
      "grade": "9"
    },
    {
      "id": "550e8400-e29b-41d4-a716-446655440022",
      "gradingTypeId": "550e8400-e29b-41d4-a716-446655440010",
      "gradingTypeName": "Regular",
      "gradingCompanyId": "550e8400-e29b-41d4-a716-446655440000",
      "gradingCompanyName": "PSA",
      "grade": "8"
    }
  ],
  "total": 10,
  "gradingType": {
    "id": "550e8400-e29b-41d4-a716-446655440010",
    "gradingCompanyId": "550e8400-e29b-41d4-a716-446655440000",
    "gradingCompanyName": "PSA",
    "name": "Regular"
  },
  "gradingCompany": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "name": "PSA",
    "description": "Professional Sports Authenticator",
    "note": null
  }
}
```

## Response Fields

### GradingCompaniesResponse

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| companies | array | Yes | Array of GradingCompany objects |
| total | number | Yes | Total number of grading companies |

### GradingTypesResponse

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| types | array | Yes | Array of GradingType objects |
| total | number | Yes | Total number of grading types for this company |
| gradingCompany | object | Yes | Parent grading company information |

### GradesResponse

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| grades | array | Yes | Array of Grade objects |
| total | number | Yes | Total number of grades for this type |
| gradingType | object | Yes | Parent grading type information |
| gradingCompany | object | Yes | Grandparent grading company information |

### GradingCompany Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| id | UUID | Yes | Unique grading company identifier |
| name | string | Yes | Company name (e.g., "PSA", "BGS", "SGC", "CGC") |
| description | string | No | Detailed description of the company |
| note | string | No | Additional notes or important information |

### GradingType Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| id | UUID | Yes | Unique grading type identifier |
| gradingCompanyId | UUID | Yes | ID of the parent grading company |
| gradingCompanyName | string | Yes | Name of the parent company |
| name | string | Yes | Type name (e.g., "Regular", "Authentic", "Black Label") |

### Grade Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| id | UUID | Yes | Unique grade identifier |
| gradingTypeId | UUID | Yes | ID of the parent grading type |
| gradingTypeName | string | Yes | Name of the parent grading type |
| gradingCompanyId | UUID | Yes | ID of the grading company |
| gradingCompanyName | string | Yes | Name of the grading company |
| grade | string | Yes | Grade value (e.g., "10", "9.5", "9", "Authentic") |

## Common Use Cases
- Build hierarchical grade selector (Company → Type → Grade)
- Display available grading companies and their services
- Look up grade IDs for use in collection card entries
- Understand grading types offered by each company (Regular, Black Label, etc.)

## Tips for AI Assistants
- The SDK returns `{ data, error }` - always check for errors first
- API is hierarchical: Companies → Types → Grades (3 levels)
- You must navigate through the hierarchy: get companies first, then types, then grades
- The `grade` field is a string to support decimal grades ("9.5") and special values ("Authentic")
- Each response includes parent information for context (e.g., GradesResponse includes both gradingType and gradingCompany)
- Use the grade `id` when adding graded cards to a collection
- Common companies: PSA, BGS (Beckett), SGC, CGC
- Grading types vary by company (e.g., PSA has "Regular" and "Authentic", BGS has "Regular" and "Black Label")
