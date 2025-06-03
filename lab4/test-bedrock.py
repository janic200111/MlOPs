import os
import json
import boto3


def main():
    # 1. Region (must be Bedrock‐enabled)
    region = "us-east-1"
    print(f"Using AWS region: {region}")

    # 2. Bedrock Runtime client
    runtime_client = boto3.client("bedrock-runtime", region_name=region)

    # 3. Model ID (chat‐style)
    model_id = "amazon.nova-micro-v1:0"

    # 4. Build a proper "messages" payload
    prompt_text = "Write a friendly greeting to a new user, using a pirate tone."
    body_payload = {
        "inferenceConfig": {"max_new_tokens": 1000},
        "messages": [{"role": "user", "content": [{"text": prompt_text}]}],
    }
    # 5. Invoke
    response = runtime_client.invoke_model(
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body_payload).encode("utf-8"),
    )

    # 6. Parse response
    body_bytes = response["body"].read()
    obj = json.loads(body_bytes.decode("utf-8"))

    # 7) Print the full JSON for inspection (optional)
    print("\n=== Full Response JSON ===")
    print(json.dumps(obj, indent=2))

    # 8) Extract generated text from obj["output"]["message"]["content"][0]["text"]
    try:
        generated_text = obj["output"]["message"]["content"][0]["text"]
    except (KeyError, IndexError):
        print(
            "\nERROR: Could not find generated text at ['output']['message']['content'][0]['text']"
        )
        return

    print("\n=== Extracted Generated Text ===")
    print(generated_text.strip())


if __name__ == "__main__":
    main()
