https://api.ximilar.com/collectibles/v2/sport_id      → sport cards




request

curl https://api.ximilar.com/collectibles/v2/sport_id -H "Content-Type: application/json" -H "Authorization: Token __API_TOKEN__" -d '{
  "records": [
      {
        "_url": "__PATH_TO_IMAGE_URL__"
      }
  ],
  "pricing": true,
  "magic_ai": true
}'


response

{
  "records": [
    {
      "_url": "https://images.ximilar.com/examples/demo/collectibles/jirip.jpeg",
      "_status": {
        "code": 200,
        "text": "OK",
        "request_id": "cf82301a-46a5-4648-8909-a4894ece97c6"
      },
      "_id": "0aa7edf6-2495-4d0e-8502-f8dd7da8e8c7",
      "_width": 1200,
      "_height": 1600,
      "Category": "Card/Sport Card",
      "_objects": [
        {
          "name": "Card",
          "id": "76fa9ec3-e0d9-408a-b582-55a1cd6712e0",
          "bound_box": [
            77,
            97,
            1096,
            1464
          ],
          "prob": 0.8502776026725769,
          "area": 0.7255067708333334,
          "Top Category": [
            {
              "id": "8ae26c4a-ae79-4c01-9b54-ac4e2b42e914",
              "name": "Card",
              "prob": 1.0
            }
          ],
          "_tags": {
            "Category": [
              {
                "name": "Card/Sport Card",
                "prob": 1.0,
                "id": "a5634621-4a37-4b37-aa3d-720b2d6b35ec",
                "pre-filled": true
              }
            ],
            "Side": [
              {
                "prob": 0.97608,
                "name": "front",
                "id": "651c8141-2b18-479b-a8b1-b959bc34b729"
              }
            ],
            "Subcategory": [
              {
                "prob": 0.84972,
                "name": "MMA",
                "id": "87e0dd9b-68f2-42f2-9283-2379c43f790b"
              }
            ],
            "Autograph": [
              {
                "prob": 0.97988,
                "name": "not signed",
                "id": "5c1bb98b-cd69-42c9-ab2a-75e480ffa2b0"
              }
            ],
            "Foil/Holo": [
              {
                "prob": 0.99476,
                "name": "Foil/Holo",
                "id": "8d0a5c67-eb08-4908-81c9-60e74d7028e7"
              }
            ],
            "Graded": [
              {
                "prob": 0.97944,
                "name": "no",
                "id": "3a532911-7d7c-4b9b-8442-f0f293952be6"
              }
            ]
          },
          "_tags_simple": [
            "Card/Sport Card",
            "front",
            "MMA",
            "not signed",
            "Foil/Holo"
          ],
          "_identification": {
            "best_match": {
              "year": "2021",
              "name": "Jiri Prochazka",
              "set_name": "Chronicles",
              "card_type": "Rookie Card",
              "card_number": "128",
              "subcategory": "MMA",
              "sub_set": "UFC",
              "company": "Panini",
              "full_name": "Jiri Prochazka 2021 #128 Panini Chronicles UFC",
              "links": {
                "ebay.com": "https://www.ebay.com/sch/i.html?_nkw=Jiri+Prochazka+2021+Chronicles+UFC+%23128+RC&_sacat=212",
                "comc.com": "https://www.comc.com/Cards,=Jiri+Prochazka+2021+Chronicles+UFC+%23128+RC",
                "beckett.com": "https://marketplace.beckett.com/search_new/?term=Jiri+Prochazka+2021+Chronicles+UFC+%23128+RC"
              },
              "pricing": {
                "list": [
                  {
                    "item_id": "v1|167302788807|0",
                    "item_link": "https://www.ebay.com/itm/167302788807",
                    "name": "SILVER RC Jiri Prochazka 2021 Panini Chronicles UFC Certified RC #128 MMA 📈",
                    "price": 5.0,
                    "currency": "USD",
                    "country_code": "US",
                    "source": "eBay",
                    "date_of_creation": "2025-02-08",
                    "date_of_sale": null,
                    "grade_company": null,
                    "grade": null,
                    "version": "Foil/Holo",
                    "variation": null
                  }
                ]
              }
            },
            "alternatives": [
              {
                "year": "2021",
                "name": "Jiri Prochazka",
                "set_name": "Chronicles",
                "card_type": "Rookie Card",
                "card_number": "128",
                "sub_set": "UFC",
                "subcategory": "MMA",
                "company": "Panini",
                "full_name": "Jiri Prochazka 2021 #128 Panini Chronicles UFC",
                "links": {
                  "ebay.com": "https://www.ebay.com/sch/i.html?_nkw=Jiri+Prochazka+2021+Chronicles+UFC+%23128+RC&_sacat=212",
                  "comc.com": "https://www.comc.com/Cards,=Jiri+Prochazka+2021+Chronicles+UFC+%23128+RC",
                  "beckett.com": "https://marketplace.beckett.com/search_new/?term=Jiri+Prochazka+2021+Chronicles+UFC+%23128+RC"
                }
              }
            ],
            "distances": [
              0.20778613,
              0.33050093
            ]
          }
        }
      ],
      "Graded Slab": [
        {
          "prob": 0.97944,
          "name": "no",
          "id": "3a532911-7d7c-4b9b-8442-f0f293952be6"
        },
        {
          "prob": 0.02056,
          "name": "yes",
          "id": "9d7bf709-dfbc-4595-91be-a9d779b5f33c"
        }
      ]
    }
  ],
  "pricing": true,
  "status": {
    "code": 200,
    "text": "OK",
    "request_id": "cf82301a-46a5-4648-8909-a4894ece97c6",
    "proc_id": "3b7a5c94-2ebb-4586-ad72-859bae9ac8a8"
  },
  "statistics": {
    "processing time": 1.959585428237915
  }
}



Details

POST
/v2/sport_id
Sports Card Identification
Taxonomy
Given a list of image records, this method identifies the largest Sport card detected in the image.

It can recognize cards from various sports (Subcategory), including Baseball, Basketball, Football, Hockey, Soccer, and MMA. The service can also detect visual features such as Foil/Holo and Autograph, and returns details like name, team, year, card_number, set_name, and company. It also provides links to relevant marketplaces, such as ebay.com, comc.com, and beckett.com.

Our sports card database includes several million cards and is growing every day, but it may not cover every set or category. We recommend to use the magic_ai option to get the best results. If you need a specific set or category added, please contact us at tech@ximilar.com.

Required attributes
Name
records
Type
dict
Max
Maximum:10
Description
A batch of JSON records (max 10). Each record represents a single image, defined by _url or _base64.

Optional attributes
Name
magic_ai
Type
boolean
Default
Default:false
Description
Not every card can be matched with our database. If set to true, the system will use our advanced AI model for card identification. This will also consume additional API credits according to the magic_ai operation. This option works with maximum 2 records/images per request. No credits will be charged if the system do not use our advanced AI model and the result is matched with our database. We recommend to send two records/images of the card with front and back side to get the best result. Or one image with both sides of the card visible similar to this IMAGE. false by default.

Name
slab_id
Type
boolean
Default
Default:false
Description
If set to true, the system will use OCR and AI to identify the slab label of the largest detected card. One slab label will be analyzed per image. This action consumes extra API credits as per the /v2/slab_id endpoint pricing. false by default.

Name
slab_grade
Type
boolean
Default
Default:false
Description
If set to true, the AI system will analyze the grading company and grade. This will also consume additional API credits according to /v2/slab_grade. false by default.

Name
analyze_all
Type
boolean
Default
Default:false
Max
Maximum:1 record/image
Description
If set to true, all cards in the image will be analyzed (only 1 record/image is allowed). This is useful for cases like a card binder. Credits are charged per each analysed card in the image, not per record (total credits = number of cards × credit price of the endpoint). false by default.

Name
pricing
Type
boolean
Default
Default:false
Description
If set to true, the system will try to return the prices. This will also consume additional API credits according to the pricing operation. No credits will be charged if the system returns no prices. false by default.

Returns
HTTP error code 2XX, if the method was OK, and other HTTP error code, if the method failed. The response body is a JSON object (map) with the following fields:

Name
records
Type
dict
Description
JSON array with the input records, each record including the field _objects (either Slab Label or Card). The main object contains _identification with detailed information about the identified card.

Name
status
Type
dict
Description
A JSON object showing the status of the request.
It includes:

code: a numeric status code (similar to HTTP status codes)
text: a short description of the status


API TOKEN 

ca9294d44b600888cb0576d443f063ab9b3fac65

Sports Cards Identification Fields
Provided for /collectibles/v2/sport_id endpoint right now for six sports.

Game	Fields
Baseball	
name, full_name, year, card_number, set_name, sub_set, subcategory, company, links

Basketball	
name, full_name, year, card_number, set_name, sub_set, subcategory, company, links

Hockey	
name, full_name, year, card_number, set_name, sub_set, subcategory, company, links

Football	
name, full_name, year, card_number, set_name, sub_set, subcategory, company, links

Soccer	
name, full_name, year, card_number, set_name, sub_set, subcategory, company, links

MMA	
name, full_name, year, card_number, set_name, sub_set, subcategory, company, links





Categories of Card	Features
Card/Sport Card	Autograph, Color, Foil/Holo, Parallel, Rotation, Side, Subcategory


features of Card/Sport Card	tags of given feature
Side	back, front
Autograph	not signed, signed
Foil/Holo	Foil/Holo, Non-Foil
Color	B/Y/G, G/W/P, R/B/P, R/W/B, aqua, black, black gold, black white, blue, boardwalk blue tiger stripe, cherry blossom, copper, frozenfactor, gold, grayscale, green, honeycomb, nebula, negative, orange, other, pink, purple, red, red gold, red green, sepia, silver, tie dye, tiger stripe, white, yellow, zebra
Parallel	camo, checker, choice, diamond, flash/shock, glitter, go space, hyper/prism, ice/atomic, icons, lava, logofractor, millionaire, mini-diamond, mojo/line, mozaic, no huddle/fast break, other, pandora, panini prizm, power/x-factor, press proof, press proof hyper, pulsar/sonar/spectrum, question mark, ray wave, red dice, shimmer, snakeskin, sonic, sparkle/speckle, stars, velocity, wave
Rotation	rotated_clockwise, rotated_counter_clockwise, rotated_upside_down, rotation_ok
Subcategory	Baseball, Basketball, Football, Hockey, MMA, Soccer, other




