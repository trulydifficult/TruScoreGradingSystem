# Using the CardSight AI API - Pricing Data API

## Overview
Get completed sale prices for any card — real market data, not opinions. Auction results reflect what buyers actually paid (bid). Buy It Now records reflect seller asking prices (ask). Together, they give you the spread. Results are grouped into raw (ungraded) and graded sections, with graded data organized by company and grade. Every record includes the source, price, URL, image, and parallel variant info. Currently available for Baseball (beta), with additional sports rolling out soon.

## Endpoint Details

### Single Card
- **Method**: GET
- **URL**: `https://api.cardsight.ai/v1/pricing/{card_id}`
- **Authentication**: API Key required (X-Api-Key header)

### Bulk Pricing (up to 100 cards)
- **Method**: POST
- **URL**: `https://api.cardsight.ai/v1/pricing/`
- **Authentication**: API Key required (X-Api-Key header)
- **Content-Type**: `application/json`

## Parameters (Single Card)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| card_id | UUID | Yes | Card UUID (path parameter) |
| period | string | No | Lookback period. Examples: "7d", "14d", "2w", "3m", "1y", "all". Omit or "all" for no time limit. |
| listing_type | string | No | Filter by listing type: "auction" (completed auctions), "fixed" (buy-it-now), or "both" (default) |
| parallel_id | UUID | No | Filter by parallel variant. Pass UUID for a specific parallel, "null" for base card only, or omit for all variants. |
| grade_id | UUID | No | Filter by grade. Pass UUID for a specific grade, "null" for ungraded only, or omit for all grades. |
| limit | integer | No | Maximum number of records to return |

## Parameters (Bulk Pricing - Request Body)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| card_ids | string[] | Yes | Array of card UUIDs (1-100). Matches the max page size of catalog search results. |
| period | string | Yes | Lookback period. Examples: "7d", "14d", "2w", "3m", "1y", "all". |
| listing_type | string | Yes | "auction", "fixed", or "both" |
| parallel_id | UUID | No | Filter by parallel variant |
| grade_id | UUID | No | Filter by grade |
| limit | integer | No | Maximum number of records per card |

## Using the CardSight AI SDK (Recommended)

### Node.js / TypeScript
```typescript
import { CardSightAI } from 'cardsightai'

const client = new CardSightAI({ apiKey: 'your-api-key' })

// Get pricing for a single card (last 90 days, all listing types)
const response = await client.pricing.get('23084701-7511-4aa3-8831-8dbddf4c2c8d', {
  period: '90d',
  listing_type: 'both'
})

// Always check for errors
if (response.error) {
  console.error('Error:', response.error)
} else {
  const data = response.data

  // Card context
  console.log(`${data.card.name} #${data.card.number}`)
  console.log(`${data.card.set.year} ${data.card.set.release}`)

  // Raw (ungraded) sales
  console.log(`\nRaw sales: ${data.raw.count} over ${data.raw.period_days} days`)
  for (const record of data.raw.records) {
    const type = record.listing_type === 'auction' ? 'Auction' : 'BIN'
    console.log(`  ${type}: $${record.price.toFixed(2)} on ${record.date}`)
    console.log(`    ${record.title}`)
    if (record.url) console.log(`    ${record.url}`)
  }

  // Graded sales (grouped by company and grade)
  for (const company of data.graded) {
    console.log(`\n${company.company_name}:`)
    for (const grade of company.grades) {
      console.log(`  Grade ${grade.grade_value}: ${grade.count} sales`)
      for (const record of grade.records) {
        console.log(`    $${record.price.toFixed(2)} on ${record.date}`)
      }
    }
  }

  // Parallel awareness — each record tells you the variant
  const parallels = data.raw.records.filter(r => r.parallel_id)
  if (parallels.length > 0) {
    console.log('\nParallel sales found:')
    for (const r of parallels) {
      console.log(`  ${r.parallel_name}: $${r.price.toFixed(2)}`)
    }
  }

  console.log(`\nTotal records: ${data.meta.total_records}`)
  console.log(`Last sale: ${data.meta.last_sale_date}`)
}

// Bulk pricing — up to 100 cards in a single request
const bulkResponse = await client.pricing.bulk({
  card_ids: [
    '23084701-7511-4aa3-8831-8dbddf4c2c8d',
    '550e8400-e29b-41d4-a716-446655440000'
  ],
  period: '90d',
  listing_type: 'both'
})

if (!bulkResponse.error) {
  const bulk = bulkResponse.data
  console.log(`${bulk.meta.successful}/${bulk.meta.requested} cards succeeded`)

  for (const result of bulk.results) {
    if (result.success && result.data) {
      console.log(`${result.data.card.name}: ${result.data.meta.total_records} records`)
    } else if (result.error) {
      console.log(`Card ${result.card_id} failed: ${result.error.message}`)
    }
  }
}
```

### Python
```python
from cardsightai import CardSightAI

client = CardSightAI(api_key='your-api-key')

# Single card pricing
response = client.pricing.get(
    '23084701-7511-4aa3-8831-8dbddf4c2c8d',
    period='90d',
    listing_type='both'
)

print(f"{response.card.name} #{response.card.number}")
print(f"Raw sales: {response.raw.count}")

for record in response.raw.records:
    listing_type = "Auction" if record.listing_type == "auction" else "BIN"
    print(f"  {listing_type}: ${record.price:.2f} on {record.date}")

# Bulk pricing
bulk = client.pricing.bulk(
    card_ids=['23084701-7511-4aa3-8831-8dbddf4c2c8d'],
    period='90d',
    listing_type='both'
)

for result in bulk.results:
    if result.success:
        print(f"{result.data.card.name}: {result.data.meta.total_records} records")
```

## Direct API Call (cURL)

### Single Card
```bash
curl -X GET "https://api.cardsight.ai/v1/pricing/23084701-7511-4aa3-8831-8dbddf4c2c8d?period=90d&listing_type=both" \
  -H "X-Api-Key: your-api-key"
```

### Bulk Pricing
```bash
curl -X POST "https://api.cardsight.ai/v1/pricing/" \
  -H "X-Api-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "card_ids": ["23084701-7511-4aa3-8831-8dbddf4c2c8d"],
    "period": "90d",
    "listing_type": "both"
  }'
```

## Example Response (Single Card)
```json
{
  "card": {
    "card_id": "23084701-7511-4aa3-8831-8dbddf4c2c8d",
    "name": "Shohei Ohtani",
    "number": "US1",
    "set": {
      "set_id": "550e8400-e29b-41d4-a716-446655440002",
      "name": "Base Set",
      "year": "2018",
      "release": "2018 Topps Update"
    },
    "parallel": null
  },
  "query": {
    "parallel_id": null,
    "grade_id": null,
    "period": "90d",
    "listing_type": "both",
    "as_of_date": "2026-03-31"
  },
  "raw": {
    "period_days": 90,
    "count": 25,
    "records": [
      {
        "title": "2018 Topps Update #US1 Shohei Ohtani RC Rookie Card",
        "price": 12.50,
        "date": "2026-03-15T00:00:00Z",
        "source": "ebay",
        "listing_type": "auction",
        "url": "https://www.ebay.com/itm/...",
        "image_url": "https://i.ebayimg.com/...",
        "parallel_id": null,
        "parallel_name": null
      },
      {
        "title": "2018 Topps Update US1 Ohtani Rookie BIN",
        "price": 18.99,
        "date": "2026-03-10T00:00:00Z",
        "source": "ebay",
        "listing_type": "fixed",
        "url": "https://www.ebay.com/itm/...",
        "image_url": "https://i.ebayimg.com/...",
        "parallel_id": null,
        "parallel_name": null
      }
    ]
  },
  "graded": [
    {
      "company_name": "PSA",
      "company_id": "550e8400-e29b-41d4-a716-446655440020",
      "grades": [
        {
          "grade_value": "10",
          "grade_id": "550e8400-e29b-41d4-a716-446655440030",
          "period_days": 90,
          "count": 8,
          "records": [
            {
              "title": "2018 Topps Update #US1 Shohei Ohtani RC PSA 10",
              "price": 85.00,
              "date": "2026-03-12T00:00:00Z",
              "source": "ebay",
              "listing_type": "auction",
              "url": "https://www.ebay.com/itm/...",
              "image_url": "https://i.ebayimg.com/...",
              "parallel_id": null,
              "parallel_name": null
            }
          ]
        }
      ]
    }
  ],
  "meta": {
    "sources": [
      { "source": "ebay", "count": 25 }
    ],
    "last_sale_date": "2026-03-15",
    "total_records": 25
  }
}
```

## Example Response (Bulk Pricing)
```json
{
  "results": [
    {
      "card_id": "23084701-7511-4aa3-8831-8dbddf4c2c8d",
      "success": true,
      "data": { "...same structure as single card response..." }
    },
    {
      "card_id": "550e8400-e29b-41d4-a716-446655440099",
      "success": false,
      "error": {
        "code": "NOT_FOUND",
        "message": "Card not found"
      }
    }
  ],
  "meta": {
    "requested": 2,
    "successful": 1,
    "failed": 1
  }
}
```

## Response Fields

### Top-Level Response

| Field | Type | Description |
|-------|------|-------------|
| card | object | Card context information |
| query | object | Echo of the query parameters that were applied |
| raw | object | Ungraded/raw pricing data |
| graded | array | Graded pricing data grouped by company and grade |
| meta | object | Response metadata |

### Card Context Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| card_id | UUID | Yes | Card UUID |
| name | string | Yes | Card name/subject |
| number | string | No | Card number in set |
| set | object | Yes | Set context: set_id, name, year, release |
| parallel | object | No | Parallel context (parallel_id, name) if filtered by parallel |

### Pricing Record Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| price | number | Yes | Sale price in USD |
| source | string | Yes | Data source (e.g., "ebay") |
| title | string | No | Listing title from marketplace |
| date | string | No | Sale date in ISO 8601 format |
| listing_type | string | No | "auction" (completed auction) or "fixed" (buy-it-now) |
| url | string | No | URL to the original listing |
| image_url | string | No | Primary image URL for the listing |
| parallel_id | UUID | No | Parallel variant UUID. Null for base card. |
| parallel_name | string | No | Parallel variant name. Null for base card. |

### Raw Section Object

| Field | Type | Description |
|-------|------|-------------|
| period_days | number/null | Period in days that was applied |
| count | number | Number of records |
| records | array | Array of PricingRecord objects |

### Graded Section

| Field | Type | Description |
|-------|------|-------------|
| company_name | string | Grading company name (e.g., "PSA") |
| company_id | UUID | Grading company UUID |
| grades | array | Array of grade groups |

### Grade Group

| Field | Type | Description |
|-------|------|-------------|
| grade_value | string | Grade value (e.g., "10", "9.5") |
| grade_id | UUID | Grade UUID |
| period_days | number/null | Period in days that was applied |
| count | number | Number of records in this group |
| records | array | Array of PricingRecord objects |

### Meta Object

| Field | Type | Description |
|-------|------|-------------|
| sources | array | Breakdown by data source (source name + count) |
| last_sale_date | string/null | Date of the most recent sale |
| total_records | number | Total records returned across all sections |

## Common Use Cases
- Show users what a card has actually sold for — let them decide what it's worth
- Compare auction prices (what the market will bear) vs Buy It Now (what sellers are asking)
- Build pricing charts showing bid/ask spread over time
- Price a full page of search results using the bulk endpoint (up to 100 cards)
- Compare base card prices to refractors and numbered parallels using parallel_id/parallel_name
- Power insurance valuations, deal-checking tools, or collection value tracking
- Filter to specific grades (e.g., PSA 10 only) for targeted price analysis

## Tips for AI Assistants
- The SDK returns `{ data, error }` — always check for errors first
- Think of auction records as "bid" (what buyers paid) and fixed as "ask" (what sellers asked)
- Period uses format: number + unit suffix — "7d", "2w", "3m", "1y", "all". Do NOT pass plain numbers like "90"
- Every record includes `parallel_id` and `parallel_name` — no extra API calls needed to identify variants
- Results are split into `raw` (ungraded) and `graded` (by company/grade) sections
- The `graded` array may be empty if no graded sales exist for the given filters
- Bulk endpoint processes each card independently — one bad ID won't affect the others
- Bulk endpoint accepts 1-100 card IDs, matching the max page size of catalog search endpoints
- `parallel_id: null` in a record means it's a base card listing
- Pass `parallel_id: "null"` as a query param to filter to base card only (string "null", not omitted)
- Coverage is expanding — the AI matching algorithm improves over time, so more listings will surface automatically
- Currently in beta for Baseball, with additional sports coming soon
