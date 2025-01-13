import argparse
import json
import os

import torch
import transformers

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="classify generated caption into visually detectable attributes and categories using Llama model.",
    )
    parser.add_argument(
        "-c",
        "--captions",
        type=str,
        default="../data/interpret/caps_batch0.json",
        help="The json file containing captions for a batch of images for all the layers. the file is a dictionary with layers_no as the keys and each value is a list of n captions, with n being the batch size",
    )
    parser.add_argument(
        "-t",
        "--token",
        type=str,
        default="",
        help="The access token for loading the model.",
    )
    parser.add_argument(
        "-m",
        "--model_path",
        type=str,
        default="../models/Meta-Llama-3.1-70B-Instruct",  # can replace with 8B model for gpu restricted setups
        help="The name of the Llama model used for interpretation and classification.",
    )

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="../data/interpret",
        help="The output path for the classification results.",
    )

    args = parser.parse_args()

    model_path = args.model_path
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path, local_files_only=True
    )

    pipeline = transformers.pipeline(
        "text-generation",
        token=args.token,
        model=model_path,
        tokenizer=tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    batch_name = os.path.basename(args.captions)
    with open(args.captions) as f:
        d = json.load(f)

    labeled_caps = dict()

    # getting the names of all layers in the json dictionary
    layers = list(d.keys())
    for layer in layers:
        curr_layer = []
        for i, sample in enumerate(d[layer]):
            # propmt:
            messages = [
                {
                    "role": "system",
                    "content": """
            Label Visual Attributes in Captions

            You help me label captions with visual attributes. I give you a caption in my prompt.
            For each part of the caption you give upto 3 labels. here is the list of possible labels:
            [male,female,person,young,old,facial expression,hair color,hair style,body part,object,furniture,vehicle,electronics,tools,toys,shape,
            texture,color,size,material,pattern,composition,style,position,state,orientation,surface,background,clothing,accessories,atmosphere,event,
            outdoor scene,indoor scene,nature,plants,body of water,weather, season,buildings,lighting,food,drink,dessert,fruits,sport,transportation,signs,animal,
            locations or places,rooms,actions,interactions,relations,technology,numbers,time,UNKNOWN].
            before assigning the class, remove non-words. an example of non-word is "'s" and "-".
            Do not remove the repeated words.
            If you do not know the word label it as "UNKNOWN". It is better to use "UNKNOWN" then to put labels you are unsure or not confident about!
            You do not give any explanations.
            Respond with a json dictionary where the keys are the parts of the caption and the values are 3 classes per each part of the caption.
            """,
                },
                {
                    "role": "user",
                    "content": "The girl with the yellow sun hat is holding a cat.",
                },
                {
                    "role": "assistant",
                    "content": """
            {
            "The girl": ["person", "young", "female"],
            "yellow": ["color", "", ""],
            "sun hat": ["clothing","season","object"],
            "is holding": ["action","",""],
            "a cat": ["animal","",""]
            }
                """,
                },
                {
                    "role": "user",
                    "content": "the handgadragon girl 's fat with dragon 's adobe handcgadragon 's handgadragon 's handcgadragon 's ' sports ... ",
                },
                {
                    "role": "assistant",
                    "content": """
                {
            "the handgadragon" : ["UNKNOWN","",""]
            "girl": ["person","female","young"],
            "fat": ["state","",""],
            "with dragon": ["animal","",""],
            "adobe": ["UNKNOWN","",""],
            "handcgadragon": ["UNKNOWNn","",""],
            "handgadragon": ["UNKNOWN","",""],
            "handcgadragon": ["UNKNOWNn","",""],
            "sports": ["sport","",""]
        }
            """,
                },
                {"role": "user", "content": sample},
            ]

            outputs = pipeline(messages, max_new_tokens=1024, do_sample=True, top_k=1)
            o = outputs[0]["generated_text"][-1]["content"]
            o = o.replace("\n", "")
            curr_layer.append(o)
            print(
                f"-----------------sample {i} in {layer} labeled!---------------------------"
            )

        with open(
            os.path.join(
                args.output,
                f"classification_llama_70B_top3_labels_{layer}_{batch_name}.json",
            ),
            "w",
        ) as ff:
            json.dump(curr_layer, ff)
        print(
            f"***************layer {layer} finished!*********************************"
        )
