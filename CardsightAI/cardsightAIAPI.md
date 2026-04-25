# Using the CardSight AI API - Card Identification API

## Overview
Identify trading cards from photos using CardSight AI's computer vision. Upload an image and get back detailed card information including player name, year, manufacturer, release, and set. The AI can detect multiple cards in a single image and automatically detects graded slabs, identifying the grading company (PSA, BGS, CGC, SGC, TAG, and others). Currently supports baseball, football, basketball, and hockey, with more sports coming soon.

## Endpoint Details
- **Default**: POST `https://api.cardsight.ai/v1/identify/card` (defaults to baseball)
- **Segment-Specific**: POST `https://api.cardsight.ai/v1/identify/card/{segment}` (e.g., `/v1/identify/card/football`)
- **Authentication**: API Key required (X-Api-Key header)
- **Content-Type**: `multipart/form-data` or direct binary (`image/jpeg`, `image/png`, `image/webp`)

> **Note:** Segment names in the URL path are case-insensitive (e.g., `football`, `Football`, and `FOOTBALL` are all valid).

## Constraints
- **Maximum file size**: 20MB
- **Supported formats**: JPEG, PNG, WebP, HEIF, HEIC

## Using the CardSight AI SDK (Recommended)

### Node.js / TypeScript
```typescript
import { CardSightAI } from 'cardsightai'
import * as fs from 'fs'

const client = new CardSightAI({ apiKey: 'your-api-key' })

// From a file path
const imageBuffer = fs.readFileSync('path/to/card-image.jpg')
const file = new File([imageBuffer], 'card.jpg', { type: 'image/jpeg' })

// Default identification (baseball)
const response = await client.identify.card(file)

// Segment-specific identification (e.g., football)
const footballResponse = await client.identify.cardBySegment('football', file)

// Always check for errors
if (response.error) {
  console.error('Error:', response.error)
} else {
  console.log('Success:', response.data.success)
  console.log('Request ID:', response.data.requestId)
  console.log('Processing Time:', response.data.processingTime, 'ms')

  // Process each detected card
  for (const detection of response.data.detections || []) {
    console.log('Confidence:', detection.confidence)

    // card is always present — completeness varies by match level
    const card = detection.card
    console.log('Year:', card.year)
    console.log('Release:', card.releaseName)

    if (card.id) {
      // Exact card match found in catalog
      console.log('Card ID:', card.id)
      console.log('Player:', card.name)
      console.log('Number:', card.number)
    }

    // Check if a graded slab was detected
    if (detection.grading) {
      console.log('Graded by:', detection.grading.company.name)
      console.log('Slab confidence:', detection.grading.confidence)
    }
  }
}
```

### Python
```python
from cardsightai import CardSightAI

client = CardSightAI(api_key='your-api-key')

# From a file path
with open('path/to/card-image.jpg', 'rb') as f:
    response = client.identify.card(f)

# Segment-specific identification (e.g., football)
with open('path/to/card-image.jpg', 'rb') as f:
    response = client.identify.card_by_segment('football', f)

print(f"Success: {response.success}")
print(f"Request ID: {response.request_id}")

for detection in response.detections:
    print(f"Confidence: {detection.confidence}")

    # card is always present — completeness varies by match level
    card = detection.card
    print(f"Year: {card.year}")
    print(f"Release: {card.release_name}")

    if card.id:
        print(f"Found: {card.name} - {card.year}")

    # Check if a graded slab was detected
    if detection.grading:
        print(f"Graded by: {detection.grading.company.name}")
        print(f"Slab confidence: {detection.grading.confidence}")
```

## Direct API Call (cURL)

### Default (Baseball)
```bash
curl -X POST "https://api.cardsight.ai/v1/identify/card" \
  -H "X-Api-Key: your-api-key" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/card-image.jpg"
```

### Segment-Specific (Football)
```bash
curl -X POST "https://api.cardsight.ai/v1/identify/card/football" \
  -H "X-Api-Key: your-api-key" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@path/to/card-image.jpg"
```

## Example Response (Exact Card Match)
```json
{
  "success": true,
  "requestId": "98cc3c7d-09a1-4433-aa9a-1939a419d411",
  "detections": [
    {
      "confidence": "High",
      "card": {
        "id": "550e8400-e29b-41d4-a716-446655440000",
        "segmentId": "660e8400-e29b-41d4-a716-446655440001",
        "releaseId": "770e8400-e29b-41d4-a716-446655440002",
        "setId": "880e8400-e29b-41d4-a716-446655440003",
        "name": "Shohei Ohtani",
        "number": "1",
        "year": "2023",
        "manufacturer": "Topps",
        "releaseName": "Topps Chrome",
        "setName": "Base Set",
        "parallel": {
          "id": "550e8400-e29b-41d4-a716-446655440001",
          "name": "Refractor",
          "numberedTo": 299
        }
      },
      "grading": {
        "confidence": "High",
        "company": {
          "id": "11bfc982-39bc-4813-99fc-70483a4dd653",
          "name": "PSA"
        }
      }
    }
  ],
  "processingTime": 1215
}
```

## Example Response (Set-Level Match with Graded Slab)
```json
{
  "success": true,
  "requestId": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "detections": [
    {
      "confidence": "Medium",
      "card": {
        "segmentId": "660e8400-e29b-41d4-a716-446655440001",
        "releaseId": "770e8400-e29b-41d4-a716-446655440002",
        "setId": "880e8400-e29b-41d4-a716-446655440003",
        "year": "2023",
        "manufacturer": "Topps",
        "releaseName": "Topps Chrome",
        "setName": "Base Set"
      },
      "grading": {
        "confidence": "Medium",
        "company": {
          "name": "BGS"
        }
      }
    }
  ],
  "processingTime": 1215
}
```

> **Note:** The `grading` field is only present when a graded slab is detected in the image. If the card is not in a slab, this field will be absent.

## Response Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| success | boolean | Yes | Whether identification completed successfully |
| requestId | string | Yes | Unique ID for tracking this request |
| detections | array | No | Array of detected cards (can be multiple) |
| processingTime | number | No | Processing time in milliseconds |

### Detection Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| confidence | string | Yes | "High" (90-100%), "Medium" (75-89%), or "Low" (50-74%) |
| card | object | Yes | Card details — completeness varies by match level |
| grading | object | No | Grading company info — only present when a graded slab is detected |

### Card Object (always present, completeness varies)

| Field | Type | Description |
|-------|------|-------------|
| id | string | UUID of identified card. Present only for exact card matches. |
| segmentId | string | UUID of the segment. Present for exact card and set-level matches. |
| releaseId | string | UUID of the release. Present for exact card and set-level matches. |
| setId | string | UUID of the set. Present for exact card and set-level matches. |
| year | string | Release year from catalog |
| manufacturer | string | Card manufacturer (Topps, Panini, etc.) |
| releaseName | string | Release/product name |
| setName | string | Set name |
| name | string | Player or subject name. Present only for exact card matches. |
| number | string | Card number. Present only for exact card matches. |
| parallel | object | Parallel variant info. Only for exact matches with an identified parallel. |

### Parallel Object (when card is a parallel variant)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| id | UUID | Yes | Unique identifier for the parallel type |
| name | string | Yes | Parallel name (e.g., "Gold Refractor", "Black Prizm") |
| description | string | No | Additional details about the parallel |
| numberedTo | number | No | Limited print run number (e.g., 299 for /299) |
| isPartial | boolean | No | True if parallel only applies to specific cards in the set |
| cards | array | No | Card UUIDs that have this parallel (only when isPartial is true) |

### Grading Object (when a graded slab is detected)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| confidence | string | Yes | How confident the AI is that a slab was detected: "High", "Medium", or "Low" |
| company | object | Yes | The grading company that graded the card |

### Company Object (inside grading)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| id | string | No | UUID of the grading company (when identified in the catalog) |
| name | string | Yes | Name of the grading company (e.g., "PSA", "BGS", "CGC", "SGC", "TAG") |

## Common Use Cases
- Identify cards from photos for inventory management
- Quick card lookup by image instead of manual search
- Batch processing of card collections
- Mobile app card scanning features
- Football, basketball, and hockey card identification using segment-specific endpoints
- Detect graded slabs and identify the grading company automatically

## Tips for AI Assistants
- The SDK returns `{ data, error }` - always check for errors first
- Multiple cards can be detected in a single image
- `card` is always present in each detection — no need to check for its existence
- Completeness of `card` indicates match level: all fields = exact match, IDs/year/release only = set-level match, empty object = no match
- There is no `aiIdentification` field — all identification data is in the `card` object
- The `confidence` field indicates match certainty: "High" (90-100%), "Medium" (75-89%), "Low" (50-74%)
- The `grading` field is only present when a graded slab is detected — always check `if (detection.grading)` before accessing
- `grading.company.name` contains the grading company name (e.g., "PSA", "BGS", "CGC", "SGC", "TAG")
- `grading.company.id` is optional — it's present when the company is recognized in the catalog
- Slab detection works automatically — no extra parameters needed
- Use `requestId` for debugging and support requests
- Compress large images before uploading for faster processing
- Default `client.identify.card(file)` uses baseball identification
- Use `client.identify.cardBySegment(segment, file)` for other sports (e.g., football, basketball)
- Segment names are case-insensitive in the URL path
