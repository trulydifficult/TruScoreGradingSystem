# Using the CardSight AI API - Marketplace Listings API

## Overview
See what's for sale right now — active auctions, live Buy It Now listings, and a direct marketplace search link for each card. Results are grouped into raw (ungraded) and graded sections, with graded data organized by company and grade. Every record includes the source, price, URL, image, and parallel variant info. Includes a marketplace search link as the last record in the raw section so users can always find more. Currently available for Baseball (beta), with additional sports rolling out soon.

## Endpoint Details
- **Method**: GET
- **URL**: `https://api.cardsight.ai/v1/marketplace/{card_id}`
- **Authentication**: API Key required (X-Api-Key header)

## Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| card_id | UUID | Yes | Card UUID (path parameter) |
| listing_type | string | No | Filter by listing type: "auction" (active auctions), "fixed" (buy-it-now), or "both" (default) |
| parallel_id | UUID | No | Filter by parallel variant. Pass UUID for a specific parallel, "null" for base card only, or omit for all variants. |
| grade_id | UUID | No | Filter by grade. Pass UUID for a specific grade, "null" for ungraded only, or omit for all grades. |
| limit | integer | No | Maximum number of records to return |

## Using the CardSight AI SDK (Recommended)

### Node.js / TypeScript
```typescript
import { CardSightAI } from 'cardsightai'

const client = new CardSightAI({ apiKey: 'your-api-key' })

// Get active marketplace listings for a card
const response = await client.marketplace.get('23084701-7511-4aa3-8831-8dbddf4c2c8d', {
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

  // Raw (ungraded) listings
  console.log(`\nActive listings: ${data.raw.count}`)
  for (const record of data.raw.records) {
    if (record.listing_type === 'search') {
      // This is the marketplace search link
      console.log(`\nSearch for more: ${record.url}`)
      continue
    }

    const type = record.listing_type === 'auction' ? 'Auction' : 'BIN'
    const bids = record.bid_count ? ` (${record.bid_count} bids)` : ''
    console.log(`  ${type}: $${record.price?.toFixed(2) ?? 'N/A'}${bids}`)
    console.log(`    ${record.title}`)
    if (record.end_date) console.log(`    Ends: ${record.end_date}`)
    if (record.url) console.log(`    ${record.url}`)
  }

  // Graded listings (grouped by company and grade)
  for (const company of data.graded) {
    console.log(`\n${company.company_name}:`)
    for (const grade of company.grades) {
      console.log(`  Grade ${grade.grade_value}: ${grade.count} listings`)
      for (const record of grade.records) {
        const type = record.listing_type === 'auction' ? 'Auction' : 'BIN'
        console.log(`    ${type}: $${record.price?.toFixed(2) ?? 'N/A'} — ${record.title}`)
      }
    }
  }

  // Parallel awareness — each record tells you the variant
  const parallels = data.raw.records.filter(r => r.parallel_id)
  if (parallels.length > 0) {
    console.log('\nParallel listings found:')
    for (const r of parallels) {
      console.log(`  ${r.parallel_name}: $${r.price?.toFixed(2) ?? 'N/A'}`)
    }
  }

  console.log(`\nTotal listings: ${data.meta.total_records}`)
}
```

### Python
```python
from cardsightai import CardSightAI

client = CardSightAI(api_key='your-api-key')

response = client.marketplace.get(
    '23084701-7511-4aa3-8831-8dbddf4c2c8d',
    listing_type='both'
)

print(f"{response.card.name} #{response.card.number}")
print(f"Active listings: {response.raw.count}")

for record in response.raw.records:
    if record.listing_type == "search":
        print(f"Search for more: {record.url}")
        continue

    listing_type = "Auction" if record.listing_type == "auction" else "BIN"
    bids = f" ({record.bid_count} bids)" if record.bid_count else ""
    print(f"  {listing_type}: ${record.price:.2f}{bids} — {record.title}")
```

## Direct API Call (cURL)
```bash
curl -X GET "https://api.cardsight.ai/v1/marketplace/23084701-7511-4aa3-8831-8dbddf4c2c8d?listing_type=both" \
  -H "X-Api-Key: your-api-key"
```

## Example Response
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
    "listing_type": "both",
    "as_of_date": "2026-03-31"
  },
  "raw": {
    "count": 15,
    "records": [
      {
        "title": "2018 Topps Update #US1 Shohei Ohtani RC Rookie Card",
        "price": 14.99,
        "source": "ebay",
        "listing_type": "fixed",
        "url": "https://www.ebay.com/itm/...",
        "image_url": "https://i.ebayimg.com/...",
        "condition": "Near Mint or Better",
        "end_date": "2026-04-15T00:00:00Z",
        "bid_count": null,
        "parallel_id": null,
        "parallel_name": null
      },
      {
        "title": "2018 Topps Update US1 Ohtani Rookie",
        "price": 5.50,
        "source": "ebay",
        "listing_type": "auction",
        "url": "https://www.ebay.com/itm/...",
        "image_url": "https://i.ebayimg.com/...",
        "condition": null,
        "end_date": "2026-04-02T00:00:00Z",
        "bid_count": 3,
        "parallel_id": null,
        "parallel_name": null
      },
      {
        "title": "Search eBay for 2018 Topps Update Shohei Ohtani US1",
        "price": null,
        "source": "ebay",
        "listing_type": "search",
        "url": "https://www.ebay.com/sch/i.html?_nkw=...",
        "image_url": null,
        "condition": null,
        "end_date": null,
        "bid_count": null,
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
          "count": 5,
          "records": [
            {
              "title": "2018 Topps Update #US1 Shohei Ohtani RC PSA 10",
              "price": 89.99,
              "source": "ebay",
              "listing_type": "fixed",
              "url": "https://www.ebay.com/itm/...",
              "image_url": "https://i.ebayimg.com/...",
              "condition": null,
              "end_date": "2026-04-20T00:00:00Z",
              "bid_count": null,
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
      { "source": "ebay", "count": 15 }
    ],
    "total_records": 15
  }
}
```

## Response Fields

### Top-Level Response

| Field | Type | Description |
|-------|------|-------------|
| card | object | Card context information |
| query | object | Echo of the query parameters that were applied |
| raw | object | Ungraded active listings |
| graded | array | Graded active listings grouped by company and grade |
| meta | object | Response metadata |

### Card Context Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| card_id | UUID | Yes | Card UUID |
| name | string | Yes | Card name/subject |
| number | string | No | Card number in set |
| set | object | Yes | Set context: set_id, name, year, release |
| parallel | object | No | Parallel context (parallel_id, name) if filtered by parallel |

### Marketplace Record Object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| title | string | Yes | Listing title |
| source | string | Yes | Marketplace source (e.g., "ebay") |
| price | number | No | Current price or starting bid in USD. Null for search links. |
| listing_type | string | No | "auction" (active auction), "fixed" (buy-it-now), or "search" (marketplace search link) |
| url | string | No | URL to the listing or search page |
| image_url | string | No | Primary image URL |
| condition | string | No | Condition description from seller |
| end_date | string | No | Listing end date in ISO 8601 format |
| bid_count | number | No | Number of bids (auctions only) |
| parallel_id | UUID | No | Parallel variant UUID. Null for base card. |
| parallel_name | string | No | Parallel variant name. Null for base card. |

### Raw Section Object

| Field | Type | Description |
|-------|------|-------------|
| count | number | Number of active listings |
| records | array | Array of MarketplaceRecord objects. The last record may be a "search" type link. |

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
| count | number | Number of listings |
| records | array | Array of MarketplaceRecord objects |

### Meta Object

| Field | Type | Description |
|-------|------|-------------|
| sources | array | Breakdown by data source (source name + count) |
| total_records | number | Total records returned across all sections |

## Common Use Cases
- Show users where to buy a specific card right now
- Compare current auction vs Buy It Now asking prices
- Build a card shopping or deal-finding experience
- Display the marketplace search link so users can always find more listings
- Filter to specific grades (e.g., PSA 10 only) for targeted shopping
- Compare base card listings to refractors and numbered parallels using parallel_id/parallel_name
- Check real-time market availability before pricing or buying decisions

## Tips for AI Assistants
- The SDK returns `{ data, error }` — always check for errors first
- The last record in `raw.records` may have `listing_type: "search"` — this is a marketplace search link, not a real listing. Handle it separately (no price, no image).
- `price` can be null on marketplace records (especially for search links)
- `bid_count` is only meaningful for auction listings — it's null for fixed price
- `end_date` tells you when a listing expires — useful for showing urgency
- `condition` is seller-provided text and may be null
- Unlike the Pricing endpoint, Marketplace has no `period` parameter — it returns what's live right now
- Every record includes `parallel_id` and `parallel_name` — no extra API calls needed to identify variants
- Results are split into `raw` (ungraded) and `graded` (by company/grade) sections
- Pass `parallel_id: "null"` as a query param to filter to base card only (string "null", not omitted)
- Coverage is expanding — the AI matching algorithm improves over time, so more listings will surface automatically
- Currently in beta for Baseball, with additional sports coming soon
