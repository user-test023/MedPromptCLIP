# -*- coding: utf-8 -*-
import requests
import json
from tqdm import tqdm
import os

#use your api URL
url = "XXXXXX"

deployment_name = "xchat4o"
url = url.replace("{deploymentName}", deployment_name)

headers = {
    "api-key": "XXXXXX",  # use your API Key
    "tppBizNo": "XXXXXX"   # use your tppBizNo
}

# f1000images fundus
category_list = [
    'Maculopathy', 'Bietti crystalline dystrophy', 'Massive hard exudates',
    'Blur fundus without Proliferative Diabetic Retinopathy', 'macular_hole',
    'Blur fundus with suspected Proliferative Diabetic Retinopathy',
    'Myelinated nerve fiber', 'Branch Retinal Vein Occlusion', 'normal',
    'Chorioretinal atrophy-coloboma', 'Optic atrophy', 'Congenital disc abnormality',
    'Pathological myopia', 'Cotton-wool spots', 'Peripheral retinal degeneration and break',
    'Central Retinal Vein Occlusion', 'Possible glaucoma', 'central_serous_chorioretinopathy',
    'Preretinal hemorrhage', 'Disc swelling and elevation', 'retinal_artery_occlusion',
    'diabetic_retinopathy Stage 1', 'Retinitis pigmentosa', 'diabetic_retinopathy Stage 2',
    'Rhegmatogenous Retinal Detachment', 'diabetic_retinopathy Stage 3',
    'Severe hypertensive retinopathy', 'Dragged Disc', 'Silicon oil in eye',
    'epiretinal_membrane', 'Tessellated fundus', 'Fibrosis', 'Vessel tortuosity',
    'Fundus neoplasm', 'Vitreous particles', 'VKH disease', 'Large optic cup',
    'Yellow-white spots-flecks', 'Laser Spots'
]

#Skincancer pathology
category_list = [
    'nontumor_skin_chondraltissue_chondraltissue','nontumor_skin_dermis_dermis','nontumor_skin_elastosis_elastosis',
    'nontumor_skin_epidermis_epidermis','nontumor_skin_hairfollicle_hairfollicle','nontumor_skin_muscle_skeletal',
    'nontumor_skin_necrosis_necrosis','nontumor_skin_nerves_nerves','nontumor_skin_sebaceousglands_sebaceousglands',
    'nontumor_skin_subcutis_subcutis','nontumor_skin_sweatglands_sweatglands','nontumor_skin_vessel_vessel',
    'tumor_skin_epithelial_bcc','tumor_skin_epithelial_sqcc','tumor_skin_melanoma_melanoma','tumor_skin_naevus_naevus',
]

#NCT
# category_list = [
#     'Adipose','Background','Cancer-associated _stroma','Colorectal_adenocarcinoma_epithelial','Debris',
#     'Lymphocytes','Mucus','Normal_colon_mucosa','Smooth_muscle',
# ]

#
# category_list = [
#     'Benign_colonic_tissue','Benign_lung_tissue','Colon _adenocarcinoma','Lung_adenocarcinoma','Lung_squamous_cell_carcinoma',]

#LungHist700
# category_list = [
#     'Moderately_differentiated_adenocarcinoma','Normal','Poorly_differentiated_adenocarcinoma','Pulmonary_squamous_cell_carcinoma_moderately_differentiatedd',
#     'Pulmonary_squamous_cell_carcinoma_poorly_differentiatedd','Pulmonary_squamous_cell_carcinoma_well_differentiatd','Well-differentiated_adenocarcinoma',
# ]

#Breast_Ultrasound
# category_list = [
#     'Benign_breast','Malignant_breast_cancer','Normal_breast_tissue',
# ]

#Breast_MRI
# category_list = [
#     'Sick_breast',
#     'Health_breast',
# ]

#MIAS
# category_list = [
#     'Architectural distortion','Asymmetry','Calcification','Circumscribed masses','Normal mammography scans','Other','Spiculated masses',
# ]

#BreakHis
# category_list = [
#     'benign_breast','malignant_breast',
# ]

#Chest_Xray
# category_list = [
#     'covid','normal','pneumonia',
# ]

#Chest_CT
# category_list = [
#     'Adenocarcinoma_chest','Large_cell_carcinoma_chest','Squamous_cell_carcinoma_chest','normal_chest',
# ]

# COVID_19
# category_list = [
#     'COVID','Lung_Opacity','Normal_lung','Viral_pneumonia',
# ]

#nerthus
# category_list = [
#     'Intestine_poor_cleanliness','Intestine_fair_cleanliness','Intestine_good_cleanliness','Intestine_excellent_cleanliness',
# ]

#Br35H
# category_list = [
#     'Non_tumorous','Tumorous',
# ]

#Knee
# category_list = [
#     'Doubtful_knee_osteoarthritis',
#     'Mild_knee_osteoarthritis',
#     'Moderate__knee_osteoarthritis',
#     'Normal_knee_osteoarthritis',
#     'Severe_knee_osteoarthritis',
# ]
# modal = 'oct retinal image'
# modal = 'fundus image'
# modal = 'Slit Lamp image'
# modal = 'SLO image'
# modal = 'FFA image'
# modal = 'pathology image'
# modal = 'ultrasound image'
# modal = 'X-ray image'
# modal = 'CT image'
# modal = 'MRI image'
modal = 'Endoscopy image'

json_name = "./MedPromptCLIP/templates/nerthus.json"

# fill your preamble text. If you don't need, set ""
preamble_text = """
The Nerthus dataset categorizes endoscopic images into four distinct bowel cleanliness levels based on the Boston Bowel Preparation Scale (BBPS) for each intestinal segment. The levels are defined as follows:

**Cleanliness Level 0:** The intestinal segment is inadequately prepared, with the presence of non-removable solid feces that completely obscure the mucosa from view, making endoscopic evaluation impossible.
**Cleanliness Level 1:** Part of the intestinal mucosa can be visualized, but other areas are obscured due to staining, residual feces, and/or the presence of opaque liquids, which significantly limits the assessment.
**Cleanliness Level 2:** There are minimal residues of staining, small fecal pieces, and/or opaque liquids. However, most of the intestinal mucosa (usually >90%) can be clearly observed, and the evaluation is considered adequate.
**Cleanliness Level 3:** The entire intestinal mucosa is clearly visible, with no residual staining, small fecal pieces, or opaque liquids, or only a small amount of transparent liquid present, resulting in the optimal assessment.

Based on these definitions, please answer the following question:
"""

# Prompt templates
prompts = [
    "Provide a concise summary of the typical {modal} findings in {category}, highlighting the most prominent pathological features (e.g., lesion margins, signal heterogeneity, calcifications) in one sentence.",
    "How would a radiologist describe the {modal} appearance of {category}, using precise clinical terminology (e.g., hypo-/hyperintense areas, contrast enhancement, cystic components) in one detailed sentence?",
    "Describe the {modal} presentation of key tissue compartments or layers in {category}, noting disruptions, thickening, or attenuation changes in a bulleted list condensed into one sentence.",
    "List the defining {modal} features of {category} in a structured single sentence, differentiating qualitative descriptors (e.g., border sharpness, necrosis, edema patterns) from quantitative metrics (e.g., lesion size in cm, Hounsfield unit values, standardized uptake values).",
    "Correlate the {modal} appearance of {category} with its underlying pathophysiology (e.g., cellular necrosis, vascular proliferation, fibrosis) in one detailed sentence.",
    "How do the {modal} findings in {category} correlate with clinical symptoms (e.g., pain, neurological deficits, respiratory compromise)? Provide a concise summary with an illustrative example in one sentence.",
    "What key distinguishing {modal} features differentiate {category} from other similar conditions (e.g., abscess vs. necrotic tumor, ischemic vs. hemorrhagic lesions)? Provide a comparative analysis in a single sentence.",
    "What temporal progression markers on {modal} imaging are characteristic of {category} (e.g., lesion growth rate, contrast uptake changes), and how do they relate to prognosis? Summarize in one sentence."
]

existing_responses = {}
if os.path.exists(json_name):
    with open(json_name, 'r', encoding='utf-8') as f:
        existing_responses = json.load(f)
    print(f"Loaded existing data from {json_name}. Found {len(existing_responses)} categories with responses.")
else:
    print(f"No existing file found at {json_name}. Starting from scratch.")

expected_responses_per_category = len(prompts) * 2

categories_to_process = []
for category in category_list:
    if category not in existing_responses or len(existing_responses[category]) < expected_responses_per_category:
        categories_to_process.append(category)
    else:
        print(f"Skipping {category}: already fully generated with {len(existing_responses[category])} responses.")

all_responses = existing_responses.copy()

for category in tqdm(categories_to_process, desc="Processing categories"):
    existing_category_results = all_responses.get(category, [])
    num_existing_prompts = len(existing_category_results) // 2
    prompts_to_process = prompts[num_existing_prompts:]

    all_result = existing_category_results.copy()

    for curr_prompt_template in prompts_to_process:
        curr_prompt = curr_prompt_template.format(modal=modal, category=category)
        if preamble_text and preamble_text.strip():
            final_prompt_content = preamble_text.strip() + "\n\n" + curr_prompt
        else:
            final_prompt_content = curr_prompt
        data = {
            "messages": [
                {
                    "role": "user",
                    "content": curr_prompt
                }
            ],
            "temperature": 0.9,
            "max_tokens": 150,
            "n": 2,
            "stream": False
        }

        try:
            response = requests.post(url, headers=headers, json=data, timeout=30)
            if response.status_code == 200:
                response_data = response.json()
                for choice in response_data.get("choices", []):
                    result = choice.get("message", {}).get("content", "").replace("\n\n", "") + "."
                    all_result.append(result)
            else:
                print(f"Failed to fetch data for prompt: {curr_prompt}")
                print(f"Status code: {response.status_code}, Response: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Request failed for {curr_prompt}: {e}")

    all_responses[category] = all_result

    with open(json_name, 'w', encoding='utf-8') as f:
        json.dump(all_responses, f, indent=4, ensure_ascii=False)
        print(f"Intermediate save: Updated JSON file with {category} at {json_name}")

with open(json_name, 'w', encoding='utf-8') as f:
    json.dump(all_responses, f, indent=4, ensure_ascii=False)
    print(f"Final JSON file saved to {json_name} with {len(all_responses)} categories.")