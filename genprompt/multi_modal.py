# -*- coding: utf-8 -*-
import requests
import json
from tqdm import tqdm
import os

url = "XXXX"

deployment_name = "xchat4o"
url = url.replace("{deploymentName}", deployment_name)


headers = {
    "api-key": "XXXXXXX",  # use your API Key
    "tppBizNo": "XXXXXXXX"   # use your tppBizNo
}

# Disease category list
category_list = [
    'Maculopathy', 'Bietti crystalline dystrophy', 'Massive hard exudates',
    'Blur fundus without Proliferative Diabetic Retinopathy', 'macular_hole',
    'Blur fundus with suspected Proliferative Diabetic Retinopathy',
    'Myelinated nerve fiber', 'Branch Retinal Vein Occlusion', 'normal',
    'Chorioretinal atrophy-coloboma', 'Optic atrophy', 'Congenital disc abnormality',
    'Pathological myopia', 'Cotton-wool spots', 'Peripheral retinal degeneration and break',
    'Central Retinal Vein Occlusion', 'Possible glaucoma', 'central_serous_chorioretinopathy',
    'Preretinal hemorrhage', 'Disc swelling and elevation',
    'diabetic_retinopathy Stage 1', 'Retinitis pigmentosa', 'diabetic_retinopathy Stage 2',
    'Rhegmatogenous Retinal Detachment', 'diabetic_retinopathy Stage 3',
    'Severe hypertensive retinopathy', 'Dragged Disc', 'Silicon oil in eye', 'Tessellated fundus', 'Fibrosis', 'Vessel tortuosity',
    'Fundus neoplasm', 'Vitreous particles', 'VKH disease', 'Large optic cup',
    'Yellow-white spots-flecks', 'Laser Spots', 'benign_myopia', 'malignant_myopia', 'cataract', 'diabetes', 'glaucoma', 'hypertension', 'myopia',
    'other diseases or abnormalities', 'age-related_macular_degeneration',
    'central_serous_retinopathy', 'choroidal_neovascularization', 'diabetic_macular_edema',
    'diabetic_retinopathy', 'drusen', 'epiretinal_membrane', 'retinal_artery_occlusion', 'retinal_vein_occlusion',
    'vitreomacular_interface_disease', 'mild_epiphype', 'moderate_epiphype', 'severe_epiphype', 'macular_edema'
]

# Modality list
modal_list = ['oct retinal', 'fundus', 'Slit Lamp', 'SLO', 'FFA']

# Output JSON file path
json_name = "./MedPromptCLIP/templates/all_eye.json"

# Prompt templates
prompt_templates = [
    "Provide a concise summary of the typical {modal} findings in {category}, highlighting the most prominent pathological features (e.g., drusen in macular degeneration, subretinal fluid in retinal detachment) in one sentence.",
    "How would an ophthalmologist describe the {modal} findings of {category} in a detailed, single sentence, using precise clinical terminology (e.g., hyperreflectivity, neovascularization, capillary non-perfusion)?",
    "Describe the {modal} appearance of each retinal layer from the internal limiting membrane (ILM) to the retinal pigment epithelium (RPE) in {category}, noting disruptions, thickenings, or signal changes (e.g., hyporeflective gaps, layer separation) in a bulleted list in one sentence.",
    "List the defining {modal} features of {category} in a structured format in single sentence, differentiating qualitative attributes (e.g., hyperreflective foci, hyporeflective cysts, leakage patterns) from quantitative parameters (e.g., central macular thickness in μm, lesion area in mm², subretinal fluid volume in mm³).",
    "Correlate the {modal} presentation of {category} with its underlying cellular pathophysiology (e.g., photoreceptor damage, Müller cell dysfunction, vascular leakage) in one detailed sentence.",
    "How do the {modal} findings in {category} correlate with clinical symptoms such as visual acuity loss, metamorphopsia, scotomas, or visual field defects? Provide a concise summary with supporting evidence (e.g., macular edema causing blurred vision) in one sentence.",
    "What are the key distinguishing {modal} features of {category} that differentiate it from similar retinal or optic nerve diseases (e.g., wet vs. dry AMD, diabetic retinopathy vs. hypertensive retinopathy)? Provide a comparative analysis in one sentence.",
    "What temporal progression markers in {modal} images are pathognomonic for {category} (e.g., increasing drusen size, subretinal fluid accumulation), and how do they correlate with prognosis? Summarize in one sentence."
]

# Step 1: Load existing results if available
existing_responses = {}
if os.path.exists(json_name):
    with open(json_name, 'r', encoding='utf-8') as f:
        existing_responses = json.load(f)
    print(f"Loaded existing data from {json_name}. Found {len(existing_responses)} categories with responses.")
else:
    print(f"No existing file found at {json_name}. Starting from scratch.")

# Step 2: Calculate expected number of responses per category (2 responses per prompt per modal)
expected_responses_per_category = len(modal_list) * len(prompt_templates) * 2

# Initialize all_responses with existing data
all_responses = existing_responses.copy()

# Step 3: Process each category
for category in tqdm(category_list, desc="Processing categories"):
    # Check if category is already fully processed
    if category in all_responses and len(all_responses[category]) >= expected_responses_per_category:
        print(f"Skipping {category}: already fully generated with {len(all_responses[category])} responses.")
        continue

    # Initialize or retrieve existing results for this category
    category_results = all_responses.get(category, [])

    # Generate prompts and responses for each modal
    for modal in modal_list:
        modal_prompts = [template.format(modal=modal, category=category) for template in prompt_templates]

        for curr_prompt in modal_prompts:
            # Skip if this prompt's responses are already present (assuming 2 responses per prompt)
            if len(category_results) >= expected_responses_per_category:
                break

            # API request payload
            data = {
                "messages": [{"role": "user", "content": curr_prompt}],
                "temperature": 0.9,
                "max_tokens": 150,
                "n": 2,
                "stop": ['.'],
                "stream": False
            }

            # Send request with retry logic
            for attempt in range(3):
                try:
                    response = requests.post(url, headers=headers, json=data, timeout=20)
                    if response.status_code == 200:
                        response_data = response.json()
                        for choice in response_data.get("choices", []):
                            content = choice.get("message", {}).get("content", "").replace("\n\n", "")
                            category_results.append(f"{content}.")
                        break
                    else:
                        if attempt == 2:
                            print(f"Failed to fetch {curr_prompt} after 3 attempts: Status {response.status_code}")
                            category_results.append("API request error.")
                except Exception as e:
                    if attempt == 2:
                        print(f"Failed to fetch {curr_prompt} after 3 attempts: {e}")
                        category_results.append("API request error.")

    # Update all_responses with this category's results
    all_responses[category] = category_results

    # Step 4: Save intermediate results after each category
    with open(json_name, 'w', encoding='utf-8') as f:
        json.dump(all_responses, f, indent=4, ensure_ascii=False)
        print(f"Intermediate save: Updated JSON file with {category} at {json_name}")

# Step 5: Final save to confirm all data
with open(json_name, 'w', encoding='utf-8') as f:
    json.dump(all_responses, f, indent=4, ensure_ascii=False)
    print(f"Final JSON file saved to {json_name} with {len(all_responses)} categories.")