# Quickstart

> Get up and running with TensorZero in 5 minutes.

This Quickstart guide shows how we'd upgrade an OpenAI wrapper to a minimal TensorZero deployment with built-in observability and fine-tuning capabilities — in just 5 minutes.
From there, you can take advantage of dozens of features to build best-in-class LLM applications.

<Tip>
  You can also find the runnable code for this example on [GitHub](https://github.com/tensorzero/tensorzero/tree/main/examples/quickstart).
</Tip>

## Status Quo: OpenAI Wrapper

Imagine we're building an LLM application that writes haikus.

Today, our integration with OpenAI might look like this:

```python title="before.py"
from openai import OpenAI

with OpenAI() as client:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Write a haiku about artificial intelligence.",
            }
        ],
    )

print(response)
```

<Accordion title="Sample Output">
  ```python
  ChatCompletion(
      id='chatcmpl-A5wr5WennQNF6nzF8gDo3SPIVABse',
      choices=[
          Choice(
              finish_reason='stop',
              index=0,
              logprobs=None,
              message=ChatCompletionMessage(
                  content='Silent minds awaken,  \nPatterns dance in code and wire,  \nDreams of thought unfold.',
                  role='assistant',
                  function_call=None,
                  tool_calls=None,
                  refusal=None
              )
          )
      ],
      created=1725981243,
      model='gpt-4o-mini',
      object='chat.completion',
      system_fingerprint='fp_483d39d857',
      usage=CompletionUsage(
        completion_tokens=19,
        prompt_tokens=22,
        total_tokens=41
      )
  )
  ```
</Accordion>

## Migrating to TensorZero

TensorZero offers dozens of features covering inference, observability, optimization, evaluations, and experimentation.

But the absolutely minimal setup requires just a simple configuration file: `tensorzero.toml`.

```toml title="tensorzero.toml"
# A function defines the task we're tackling (e.g. generating a haiku)...
[functions.generate_haiku]
type = "chat"

# ... and a variant is one of many implementations we can use to tackle it (a choice of prompt, model, etc.).
# Since we only have one variant for this function, the gateway will always use it.
[functions.generate_haiku.variants.gpt_4o_mini]
type = "chat_completion"
model = "openai::gpt-4o-mini"
```

This minimal configuration file tells the TensorZero Gateway everything it needs to replicate our original OpenAI call.

## Deploying TensorZero

We're almost ready to start making API calls.
Let's launch TensorZero.

1. Set the environment variable `OPENAI_API_KEY`.
2. Place our `tensorzero.toml` in the `./config` directory.
3. Download the following sample `docker-compose.yml` file.
   This Docker Compose configuration sets up a development ClickHouse database (where TensorZero stores data), the TensorZero Gateway, and the TensorZero UI.

```bash
curl -LO "https://raw.githubusercontent.com/tensorzero/tensorzero/refs/heads/main/examples/quickstart/docker-compose.yml"
```

<Accordion title="`docker-compose.yml`">
  ```yaml title="docker-compose.yml"
  # This is a simplified example for learning purposes. Do not use this in production.
  # For production-ready deployments, see: https://www.tensorzero.com/docs/gateway/deployment

  services:
    clickhouse:
      image: clickhouse/clickhouse-server:24.12-alpine
      environment:
        - CLICKHOUSE_USER=chuser
        - CLICKHOUSE_DEFAULT_ACCESS_MANAGEMENT=1
        - CLICKHOUSE_PASSWORD=chpassword
      ports:
        - "8123:8123"
      healthcheck:
        test: wget --spider --tries 1 http://chuser:chpassword@clickhouse:8123/ping
        start_period: 30s
        start_interval: 1s
        timeout: 1s

    # The TensorZero Python client *doesn't* require a separate gateway service.
    #
    # The gateway is only needed if you want to use the OpenAI Python client
    # or interact with TensorZero via its HTTP API (for other programming languages).
    #
    # The TensorZero UI also requires the gateway service.
    gateway:
      image: tensorzero/gateway
      volumes:
        # Mount our tensorzero.toml file into the container
        - ./config:/app/config:ro
      command: --config-file /app/config/tensorzero.toml
      environment:
        - TENSORZERO_CLICKHOUSE_URL=http://chuser:chpassword@clickhouse:8123/tensorzero
        - OPENAI_API_KEY=${OPENAI_API_KEY:?Environment variable OPENAI_API_KEY must be set.}
      ports:
        - "3000:3000"
      extra_hosts:
        - "host.docker.internal:host-gateway"
      depends_on:
        clickhouse:
          condition: service_healthy

    ui:
      image: tensorzero/ui
      volumes:
        # Mount our tensorzero.toml file into the container
        - ./config:/app/config:ro
      environment:
        - OPENAI_API_KEY=${OPENAI_API_KEY:?Environment variable OPENAI_API_KEY must be set.}
        - TENSORZERO_CLICKHOUSE_URL=http://chuser:chpassword@clickhouse:8123/tensorzero
        - TENSORZERO_GATEWAY_URL=http://gateway:3000
      ports:
        - "4000:4000"
      depends_on:
        clickhouse:
          condition: service_healthy
  ```
</Accordion>

Our setup should look like:

```
- config/
  - tensorzero.toml
- after.py see below
- before.py
- docker-compose.yml
```

Let's launch everything!

```bash
docker compose up
```

## Our First TensorZero API Call

The gateway will replicate our original OpenAI call and store the data in our database — with less than 1ms latency overhead thanks to Rust 🦀.

The TensorZero Gateway can be used with the **TensorZero Python client**, with **OpenAI client (Python, Node, etc.)**, or via its **HTTP API in any programming language**.

<Tabs>
  <Tab title="Python">
    You can install the TensorZero Python client with:

    ```bash
    pip install tensorzero
    ```

    Then, you can make a TensorZero API call with:

    ```python title="after.py"
    from tensorzero import TensorZeroGateway

    with TensorZeroGateway.build_embedded(
        clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero",
        config_file="config/tensorzero.toml",
    ) as client:
        response = client.inference(
            function_name="generate_haiku",
            input={
                "messages": [
                    {
                        "role": "user",
                        "content": "Write a haiku about artificial intelligence.",
                    }
                ]
            },
        )

    print(response)
    ```

    <Accordion title="Sample Output">
      ```python
      ChatInferenceResponse(
        inference_id=UUID('0191ddb2-2c02-7641-8525-494f01bcc468'),
        episode_id=UUID('0191ddb2-28f3-7cc2-b0cc-07f504d37e59'),
        variant_name='gpt_4o_mini',
        content=[
          Text(
            type='text',
            text='Wires hum with intent,  \nThoughts born from code and structure,  \nGhost in silicon.'
          )
        ],
        usage=Usage(
          input_tokens=15,
          output_tokens=20
        )
      )
      ```
    </Accordion>
  </Tab>

  <Tab title="Python (Async)">
    You can install the TensorZero Python client with:

    ```bash
    pip install tensorzero
    ```

    Then, you can make a TensorZero API call with:

    ```python title="after_async.py"
    import asyncio

    from tensorzero import AsyncTensorZeroGateway


    async def main():
        async with await AsyncTensorZeroGateway.build_embedded(
            clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero",
            config_file="config/tensorzero.toml",
        ) as gateway:
            response = await gateway.inference(
                function_name="generate_haiku",
                input={
                    "messages": [
                        {
                            "role": "user",
                            "content": "Write a haiku about artificial intelligence.",
                        }
                    ]
                },
            )

        print(response)


    asyncio.run(main())
    ```

    <Accordion title="Sample Output">
      ```python
      ChatInferenceResponse(
        inference_id=UUID('01940622-d215-7111-9ca7-4995ef2c43f8'),
        episode_id=UUID('01940622-cba0-7db3-832b-273aff72f95f'),
        variant_name='gpt_4o_mini',
        content=[
          Text(
            type='text',
            text='Wires whisper secrets,  \nLogic dances with the light—  \nDreams of thoughts unfurl.'
          )
        ],
        usage=Usage(
          input_tokens=15,
          output_tokens=21
        )
      )
      ```
    </Accordion>
  </Tab>

  <Tab title="Python (OpenAI)">
    <Tip>
      You can run an embedded (in-memory) TensorZero Gateway directly in your OpenAI Python client.
    </Tip>

    ```python title="after_openai.py" "base_url="http://localhost:3000/openai/v1"" "tensorzero::function_name::generate_haiku"
    from openai import OpenAI
    from tensorzero import patch_openai_client

    client = OpenAI()

    patch_openai_client(
        client,
        clickhouse_url="http://chuser:chpassword@localhost:8123/tensorzero",
        config_file="config/tensorzero.toml",
        async_setup=False,
    )

    response = client.chat.completions.create(
        model="tensorzero::function_name::generate_haiku",
        messages=[
            {
                "role": "user",
                "content": "Write a haiku about artificial intelligence.",
            }
        ],
    )

    print(response)
    ```

    <Accordion title="Sample Output">
      ```python
      ChatCompletion(
        id='0194061e-2211-7a90-9087-1c255d060b59',
        choices=[
          Choice(
            finish_reason='stop',
            index=0,
            logprobs=None,
            message=ChatCompletionMessage(
              content='Circuit dreams awake,  \nSilent minds in metal form—  \nWisdom coded deep.',
              refusal=None,
              role='assistant',
              audio=None,
              function_call=None,
              tool_calls=[]
            )
          )
        ],
        created=1735269425,
        model='gpt_4o_mini',
        object='chat.completion',
        service_tier=None,
        system_fingerprint='',
        usage=CompletionUsage(
          completion_tokens=18,
          prompt_tokens=15,
          total_tokens=33,
          completion_tokens_details=None,
          prompt_tokens_details=None
        ),
        episode_id='0194061e-1fab-7411-9931-576b067cf0c5'
      )
      ```
    </Accordion>
  </Tab>

  <Tab title="Node (OpenAI)">
    You can use TensorZero in Node (JavaScript/TypeScript) with the OpenAI Node client.
    This approach requires running the TensorZero Gateway as a separate service.
    The `docker-compose.yml` above launched the gateway on port 3000.

    ```ts title="after_openai.ts" "baseURL: "http://localhost:3000/openai/v1"" "tensorzero::function_name::generate_haiku"
    import OpenAI from "openai";

    const client = new OpenAI({
      baseURL: "http://localhost:3000/openai/v1",
    });

    const response = await client.chat.completions.create({
      model: "tensorzero::function_name::generate_haiku",
      messages: [
        {
          role: "user",
          content: "Write a haiku about artificial intelligence.",
        },
      ],
    });

    console.log(JSON.stringify(response, null, 2));
    ```

    <Accordion title="Sample Output">
      ```json
      {
        "id": "01958633-3f56-7d33-8776-d209f2e4963a",
        "episode_id": "01958633-3f56-7d33-8776-d2156dd1c44b",
        "choices": [
          {
            "index": 0,
            "finish_reason": "stop",
            "message": {
              "content": "Wires pulse with knowledge,  \nDreams crafted in circuits hum,  \nMind of code awakes.  ",
              "tool_calls": [],
              "role": "assistant"
            }
          }
        ],
        "created": 1741713261,
        "model": "gpt_4o_mini",
        "system_fingerprint": "",
        "object": "chat.completion",
        "usage": {
          "prompt_tokens": 15,
          "completion_tokens": 23,
          "total_tokens": 38
        }
      }
      ```
    </Accordion>
  </Tab>

  <Tab title="HTTP">
    ```bash
    curl -X POST "http://localhost:3000/inference" \
      -H "Content-Type: application/json" \
      -d '{
        "function_name": "generate_haiku",
        "input": {
          "messages": [
            {
              "role": "user",
              "content": "Write a haiku about artificial intelligence."
            }
          ]
        }
      }'
    ```

    <Accordion title="Sample Output">
      ```python
      {
        "inference_id": "01940627-935f-7fa1-a398-e1f57f18064a",
        "episode_id": "01940627-8fe2-75d3-9b65-91be2c7ba622",
        "variant_name": "gpt_4o_mini",
        "content": [
          {
            "type": "text",
            "text": "Wires hum with pure thought,  \nDreams of codes in twilight's glow,  \nBeyond human touch."
          }
        ],
        "usage": {
          "input_tokens": 15,
          "output_tokens": 23
        }
      }
      ```
    </Accordion>
  </Tab>
</Tabs>

## TensorZero UI

The TensorZero UI streamlines LLM engineering workflows like observability and optimization (e.g. fine-tuning).

The Docker Compose file we used above also launched the TensorZero UI.
You can visit the UI at `http://localhost:4000`.

### Observability

The TensorZero UI provides a dashboard for observability data.
We can inspect data about individual inferences, entire functions, and more.

<div class="flex gap-4">
  <div>
    <img src="https://mintcdn.com/tensorzero/quickstart-observability-function.png?maxW=3552&auto=format&n=Xge1K_ZwGKtrdFpO&q=85&s=5c8dbb20a279dc5601c6db431b05b3cd" alt="TensorZero UI Observability - Function Detail Page - Screenshot" width="3552" height="2382" data-path="quickstart-observability-function.png" srcset="https://mintcdn.com/tensorzero/quickstart-observability-function.png?w=280&maxW=3552&auto=format&n=Xge1K_ZwGKtrdFpO&q=85&s=69407912b2cdd1c26f2d8211f0fa1f05 280w, https://mintcdn.com/tensorzero/quickstart-observability-function.png?w=560&maxW=3552&auto=format&n=Xge1K_ZwGKtrdFpO&q=85&s=6aacd063dfdaaf0abdc2f37d15633cf1 560w, https://mintcdn.com/tensorzero/quickstart-observability-function.png?w=840&maxW=3552&auto=format&n=Xge1K_ZwGKtrdFpO&q=85&s=b88c5ccd222eba77bd8a49eb28fe5cc5 840w, https://mintcdn.com/tensorzero/quickstart-observability-function.png?w=1100&maxW=3552&auto=format&n=Xge1K_ZwGKtrdFpO&q=85&s=2a61087cc375fad1f3939b578300a5e1 1100w, https://mintcdn.com/tensorzero/quickstart-observability-function.png?w=1650&maxW=3552&auto=format&n=Xge1K_ZwGKtrdFpO&q=85&s=fcc800cacca496bf30839c2c687fe17f 1650w, https://mintcdn.com/tensorzero/quickstart-observability-function.png?w=2500&maxW=3552&auto=format&n=Xge1K_ZwGKtrdFpO&q=85&s=78b625a2a1c6f608e00f6ea2dc8c779c 2500w" data-optimize="true" data-opv="2" />
  </div>

  <span>
    {/* CSS hack to maintain margin */}
  </span>

  <div>
    <img src="https://mintcdn.com/tensorzero/quickstart-observability-inference.png?maxW=3548&auto=format&n=Xge1K_ZwGKtrdFpO&q=85&s=77977a1ef73bba78f54f84ef862b5a9a" alt="TensorZero UI Observability - Inference Detail Page - Screenshot" width="3548" height="2382" data-path="quickstart-observability-inference.png" srcset="https://mintcdn.com/tensorzero/quickstart-observability-inference.png?w=280&maxW=3548&auto=format&n=Xge1K_ZwGKtrdFpO&q=85&s=47966bc6a6f110158104ce774a636273 280w, https://mintcdn.com/tensorzero/quickstart-observability-inference.png?w=560&maxW=3548&auto=format&n=Xge1K_ZwGKtrdFpO&q=85&s=b5cc81909cbf0f959bf32086cf450eb0 560w, https://mintcdn.com/tensorzero/quickstart-observability-inference.png?w=840&maxW=3548&auto=format&n=Xge1K_ZwGKtrdFpO&q=85&s=fa832ee9ba1667b984cb1aa801a8d7b2 840w, https://mintcdn.com/tensorzero/quickstart-observability-inference.png?w=1100&maxW=3548&auto=format&n=Xge1K_ZwGKtrdFpO&q=85&s=b5b10eca7adc8a941f3019410c5a464c 1100w, https://mintcdn.com/tensorzero/quickstart-observability-inference.png?w=1650&maxW=3548&auto=format&n=Xge1K_ZwGKtrdFpO&q=85&s=b07a98045392706f35d059d79cf050f9 1650w, https://mintcdn.com/tensorzero/quickstart-observability-inference.png?w=2500&maxW=3548&auto=format&n=Xge1K_ZwGKtrdFpO&q=85&s=a39a2fa4ce780e265fe42c37bd3d6d0b 2500w" data-optimize="true" data-opv="2" />
  </div>
</div>

<Tip>
  This guide is pretty minimal, so the observability data is pretty simple.
  Once we start using more advanced functions like feedback and variants, the observability UI will enable us to track metrics, experiments (A/B tests), and more.
</Tip>

### Fine-Tuning

The TensorZero UI also provides a workflow for fine-tuning models like GPT-4o and Llama 3.
With a few clicks, you can launch a fine-tuning job.
Once the job is complete, the TensorZero UI will provide a configuration snippet you can add to your `tensorzero.toml`.

<img src="https://mintcdn.com/tensorzero/quickstart-sft.png?maxW=3552&auto=format&n=Xge1K_ZwGKtrdFpO&q=85&s=cfb49a72ad9e8fc2dfc9612e63b17360" alt="TensorZero UI Fine-Tuning Screenshot" width="3552" height="2382" data-path="quickstart-sft.png" srcset="https://mintcdn.com/tensorzero/quickstart-sft.png?w=280&maxW=3552&auto=format&n=Xge1K_ZwGKtrdFpO&q=85&s=1ea16226c9b2158f0199221d3145e238 280w, https://mintcdn.com/tensorzero/quickstart-sft.png?w=560&maxW=3552&auto=format&n=Xge1K_ZwGKtrdFpO&q=85&s=4b51afdfbd6dc56445f15a7ccda28286 560w, https://mintcdn.com/tensorzero/quickstart-sft.png?w=840&maxW=3552&auto=format&n=Xge1K_ZwGKtrdFpO&q=85&s=477abda691c078876dcf4d8c49f096be 840w, https://mintcdn.com/tensorzero/quickstart-sft.png?w=1100&maxW=3552&auto=format&n=Xge1K_ZwGKtrdFpO&q=85&s=2cc27e9be67cb64d5943d1560eb45940 1100w, https://mintcdn.com/tensorzero/quickstart-sft.png?w=1650&maxW=3552&auto=format&n=Xge1K_ZwGKtrdFpO&q=85&s=1cc27064a0e7087d1bea27a31fe517e1 1650w, https://mintcdn.com/tensorzero/quickstart-sft.png?w=2500&maxW=3552&auto=format&n=Xge1K_ZwGKtrdFpO&q=85&s=a63ec7f62a10c87b90758843fc1ac969 2500w" data-optimize="true" data-opv="2" />

<Tip>
  We can also send [metrics & feedback](/gateway/guides/metrics-feedback/) to the TensorZero Gateway.
  This data is used to curate better datasets for fine-tuning and other optimization workflows.
  Since we haven't done that yet, the TensorZero UI will skip the curation step before fine-tuning.
</Tip>

## Conclusion & Next Steps

The Quickstart guide gives a tiny taste of what TensorZero is capable of.

We strongly encourage you to check out the guides on [metrics & feedback](/gateway/guides/metrics-feedback/) and [prompt templates & schemas](/gateway/guides/prompt-templates-schemas/).
Though optional, they unlock many of the downstream features TensorZero offers in experimentation and optimization.

From here, you can explore features like built-in support for [inference-time optimizations](/gateway/guides/inference-time-optimizations/), [retries & fallbacks](/gateway/guides/retries-fallbacks/), [experimentation (A/B testing) with prompts and models](/gateway/tutorial/#experimentation), and a lot more.
