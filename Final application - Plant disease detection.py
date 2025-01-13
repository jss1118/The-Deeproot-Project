import tkinter as tk
import customtkinter
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import threading
import tensorflow as tf
import numpy as np
from ultralytics import YOLO
import os
import datetime
# Variable definitions
confidence = 0.5
disease = False
scale_factor = 0.5
# Folder clearing
output_dir = '/Users/joshua.stanley/Desktop/Science Research/Model outputs/R-CNN/zoomed_bounding_boxes/test_dataset/'

stopped = False

response = 'Data will be provided if detections show disease'

apple_classes = ['Apple__black_rot', 'Apple__healthy', 'Apple__rust', 'Apple__scab']
casava_classes = ['Cassava__bacterial_blight', 'Cassava__brown_streak_disease', 'Cassava__green_mottle', 'Cassava__healthy', 'Cassava__mosaic_disease']
cherry_classes = ['Cherry__healthy', 'Cherry__powdery_mildew']
chili_classes = ['Chili__healthy', 'Chili__leaf curl', 'Chili__leaf spot', 'Chili__whitefly', 'Chili__yellowish']
citrus_classes = ['Black spot', 'canker', 'greening', 'healthy']
coffee_classes = ['Coffee__cercospora_leaf_spot', 'Coffee__healthy', 'Coffee__red_spider_mite', 'Coffee__rust']
corn_classes = ['Corn__common_rust', 'Corn__gray_leaf_spot', 'Corn__healthy', 'Corn__northern_leaf_blight']
cucumber_classes = ['Cucumber__diseased', 'Cucumber__healthy']
grape_classes = ['Grape__black_measles', 'Grape__black_rot', 'Grape__healthy', 'Grape__leaf_blight_(isariopsis_leaf_spot)']
guava_classes = ['Gauva__diseased', 'Gauva__healthy']
jamun_classes  = ['Jamun__diseased', 'Jamun__healthy']
lemon_classes  = ['Lemon__diseased', 'Lemon__healthy']
mango_classes = ['Mango__diseased', 'Mango__healthy']
peach_classes = ['Peach__bacterial_spot', 'Peach__healthy']
pepper_classes = ['Pepper_bell__bacterial_spot', 'Pepper_bell__healthy']
pomegranate_classes = ['Pomegranate__diseased', 'Pomegranate__healthy']
potato_classes = ['Potato__early_blight', 'Potato__healthy', 'Potato__late_blight']
rice_classes = ['Rice__brown_spot', 'Rice__healthy', 'Rice__hispa', 'Rice__leaf_blast', 'Rice__neck_blast']
soybean_classes = ['Soybean__bacterial_blight', 'Soybean__caterpillar', 'Soybean__diabrotica_speciosa', 'Soybean__downy_mildew', 'Soybean__healthy', 'Soybean__mosaic_virus', 'Soybean__powdery_mildew', 'Soybean__rust', 'Soybean__southern_blight']
strawberry_classes = ['Strawberry___leaf_scorch', 'Strawberry__healthy']
sugarcane_classes = ['Sugarcane__bacterial_blight', 'Sugarcane__healthy', 'Sugarcane__red_rot', 'Sugarcane__red_stripe', 'Sugarcane__rust']
tea_classes = ['Tea__algal_leaf', 'Tea__anthracnose', 'Tea__bird_eye_spot', 'Tea__brown_blight', 'Tea__healthy']
tomato_classes = ['Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

classes = ['apple',
           'casava',
           'cherry',
           'chili',
           'citrus',
           'coffee',
           'corn',
           'cucumber',
           'grape',
           'guava',
           'jamun',
           'lemon',
           'mango',
           'peach',
           'pepper',
           'pomegranate',
           'potato',
           'rice',
           'soybean',
           'strawberry',
           'sugarcane',
           'tea',
           'tomato'
]
disease_dictionary = {
    "apple": [
        {
            "name": "black_rot",
            "description": "Black rot on apples is a fungal disease caused by the pathogen Botryosphaeria obtusa.\nIt often manifests as circular, brownish-black lesions on fruit and can also affect leaves and branches.\nInfection commonly occurs through wounds and is favored by warm, humid conditions.\nTo manage and prevent black rot, it’s important to remove and destroy infected plant material, prune to improve air circulation, and apply recommended fungicides during the growing season."
        },
        {
            "name": "healthy",
            "description": "A healthy apple tree shows no visible signs of disease or pest infestation.\nIts leaves are vibrant, fruit development is even, and overall growth is vigorous.\nTo maintain good health, ensure proper fertilization, adequate watering, pruning for airflow, and regular monitoring for early signs of diseases or pests.\nEmploying preventative measures—such as using resistant cultivars, practicing good orchard sanitation, and applying protective sprays as needed—helps preserve the tree’s vigor."
        },
        {
            "name": "rust (Apple Cedar Rust)",
            "description": "Apple rust diseases, such as cedar-apple rust, are caused by fungi in the genus Gymnosporangium.\nThese diseases produce bright orange or yellow spots on leaves and can cause defoliation and reduced fruit quality.\nThe fungus requires both juniper (cedar) and apple hosts to complete its life cycle.\nControlling rust involves removing nearby juniper hosts or planting resistant apple varieties.\nFungicidal sprays can be used preventatively, and good sanitation, including removing infected leaves, also helps minimize outbreaks."
        },
        {
            "name": "scab",
            "description": "Apple scab is a common fungal disease caused by Venturia inaequalis.\nIt typically appears as olive-green to black, velvety spots on leaves and fruit.\nOver time, infected leaves may yellow and drop, and fruit can become cracked or misshapen.\nPreventing apple scab involves planting resistant cultivars, applying protective fungicide sprays during key infection periods, and maintaining orchard sanitation by removing fallen leaves and pruning to improve air circulation."
        }
    ],
    "cassava": [
        {
            "name": "bacterial_blight",
            "description": "Cassava bacterial blight is caused by the bacterium Xanthomonas axonopodis pv. manihotis.\nIt creates angular leaf spots, wilting, and eventual defoliation.\nSevere infections can reduce yields significantly.\nManaging this disease involves using disease-free planting material, practicing crop rotation, and maintaining clean field conditions.\nPromptly removing and destroying infected plants can prevent the spread, and some resistant cassava varieties are available."
        },
        {
            "name": "brown_streak_disease",
            "description": "Cassava brown streak disease (CBSD) is caused by potyviruses such as Cassava brown streak virus and Ugandan cassava brown streak virus.\nInfected plants may display yellow-brown streaks on leaves and brown necrotic spots in the tubers, drastically reducing food quality and yield.\nControl strategies include using virus-free planting material, planting resistant varieties, and removing infected plants.\nEffective vector control (whiteflies) and good sanitation also help limit the disease’s impact."
        },
        {
            "name": "green_mottle",
            "description": "Green mottle in cassava typically refers to leaf discoloration or mottling caused by viral infections or nutrient imbalances.\nWhile the term is less specific than other well-known cassava diseases, the appearance of mottled leaves may signal viral presence or stress.\nManaging green mottle involves using clean planting material, ensuring adequate soil nutrition, and controlling vector insects that may spread viruses.\nGood crop hygiene and rotation practices are also beneficial."
        },
        {
            "name": "healthy",
            "description": "A healthy cassava plant has vibrant green leaves, strong stems, and well-formed tubers without any visible signs of disease.\nMaintaining plant health involves proper site selection, balanced fertilization, timely weeding, and the use of high-quality, disease-free cuttings.\nRegular monitoring for pests and diseases and implementing integrated pest management helps sustain healthy growth and good yields."
        },
        {
            "name": "mosaic_disease",
            "description": "Cassava mosaic disease (CMD) is caused by various whitefly-transmitted geminiviruses.\nIt produces yellow or pale green mosaics on leaves, stunted growth, and reduced tuber yields.\nTo control CMD, start with virus-free planting material, use resistant varieties, and remove infected plants.\nManaging whitefly populations through insecticidal soap, natural predators, and proper field hygiene is also essential for preventing the spread of the virus."
        }
    ],
    "cherry": [
        {
            "name": "healthy",
            "description": "A healthy cherry tree or fruit displays bright green leaves, robust blossoms, and cherries that mature evenly without blemishes.\nTo keep cherry trees healthy, ensure balanced fertilization, proper irrigation, and regular pruning to improve airflow and sunlight penetration.\nMonitoring for early signs of disease or pests and employing preventative fungicide or insecticide treatments when necessary helps maintain a thriving orchard."
        },
        {
            "name": "powdery_mildew",
            "description": "Powdery mildew on cherries is caused by fungal pathogens like Podosphaera clandestina.\nIt appears as white, powdery spots on leaves and young fruit, potentially leading to premature leaf drop and reduced fruit quality.\nPreventing powdery mildew involves pruning for good air circulation, avoiding over-fertilization, and applying fungicides at early infection stages.\nResistant varieties and careful orchard sanitation further help reduce disease incidence."
        }
    ],
    "chili": [
        {
            "name": "healthy",
            "description": "A healthy chili plant features dark green leaves, steady growth, and fruit that develops fully without discoloration or deformities.\nEnsuring proper soil fertility, regular watering without waterlogging, and prompt removal of weeds can help maintain plant health.\nRegular scouting for early signs of pests and diseases, along with integrated pest management, fosters strong and productive chili plants."
        },
        {
            "name": "leaf curl",
            "description": "Leaf curl in chili peppers is often caused by viruses, such as the chili leaf curl virus, transmitted by whiteflies.\nIt leads to upward or downward curling leaves, stunted growth, and reduced yields.\nManaging leaf curl involves controlling whiteflies using insecticides, reflective mulches, or natural predators, and planting disease-free seedlings.\nMaintaining good sanitation and removing infected plants also helps to prevent spread."
        },
        {
            "name": "leaf spot",
            "description": "Leaf spots on chili peppers are typically caused by fungal or bacterial pathogens.\nSymptoms include small, dark lesions on leaves that can coalesce, causing premature leaf drop and reduced fruit yield.\nPreventative measures include using certified disease-free seeds, ensuring proper spacing and airflow, and applying protective fungicides.\nRemoving infected leaves and practicing crop rotation further helps minimize outbreaks."
        },
        {
            "name": "whitefly",
            "description": "Whiteflies are tiny, sap-sucking insects that feed on the underside of chili leaves, causing yellowing, stunted growth, and soot molds due to honeydew secretions.\nControlling whiteflies involves using yellow sticky traps, introducing natural predators (like ladybugs), and carefully applying insecticidal soaps or oils.\nGood field hygiene, weed removal, and avoiding overuse of broad-spectrum insecticides help keep whitefly populations in check."
        },
        {
            "name": "yellowish",
            "description": "“Yellowish” chili plants may indicate nutrient deficiencies, water stress, or mild viral/bacterial infections.\nThe leaves may turn light green to yellow, and growth could slow.\nEnsuring balanced fertilization, proper irrigation, and good drainage can often resolve nutrient or moisture issues.\nIf a disease is suspected, removing infected parts and applying appropriate treatments or adjusting management practices helps restore plant vigor."
        }
    ],
    "citrus": [
        {
            "name": "Black spot (Citrus Black Spot)",
            "description": "Citrus black spot is a fungal disease caused by Phyllosticta citricarpa.\nIt leads to dark, speckled spots on fruit rinds, premature fruit drop, and cosmetic damage that affects market value.\nPrevention includes using certified disease-free seedlings, applying fungicidal sprays at recommended intervals, and promptly removing infected fruit from the orchard.\nGood sanitation and resistant cultivars also help control the disease."
        },
        {
            "name": "canker (Citrus Canker)",
            "description": "Citrus canker, caused by the bacterium Xanthomonas citri subsp. citri, creates raised, corky lesions with a yellow halo on leaves, stems, and fruit.\nHighly contagious, it spreads by rain, wind, and contaminated tools.\nTo control canker, implement strict sanitation measures, prune out infected tissue, use copper-based bactericides, and plant resistant varieties.\nPreventing mechanical spread by disinfecting tools and avoiding working in wet conditions also helps."
        },
        {
            "name": "greening (Citrus Greening or Huanglongbing)",
            "description": "Citrus greening (HLB) is caused by a bacterium (Candidatus Liberibacter species) spread by the Asian citrus psyllid.\nInfected trees produce misshapen, sour fruit and show mottled leaves with yellowing patterns.\nControl efforts focus on using disease-free nursery stock, controlling psyllid populations with insecticides or biocontrols, and promptly removing infected trees.\nResearch into resistant varieties and improved management practices continues."
        },
        {
            "name": "healthy (Citrus Healthy)",
            "description": "Healthy citrus trees have glossy green leaves, well-formed fruit, and vigorous growth without signs of blemishes or yellowing.\nMaintaining health involves balanced fertilization, adequate watering, and proper pruning.\nRegular scouting for pests and early diseases, coupled with integrated pest management and good orchard sanitation, ensures sustained productivity and orchard vitality."
        }
    ],
    "coffee": [
        {
            "name": "cercospora_leaf_spot",
            "description": "Cercospora leaf spot of coffee, caused by Cercospora coffeicola, presents as brownish leaf spots with a yellow halo and can also affect berries.\nSevere infestations lead to defoliation and decreased yield.\nControl includes removing infected leaves, maintaining balanced nutrition, and applying appropriate fungicides.\nUsing resistant cultivars and ensuring adequate sunlight and airflow helps reduce disease pressure."
        },
        {
            "name": "healthy",
            "description": "A healthy coffee plant boasts deep green leaves, robust branches, and well-formed coffee cherries without discoloration.\nGood soil management, consistent irrigation, balanced nutrition, and shaded growth environments can support plant health.\nRegular monitoring for pests and early disease symptoms, pruning for good air circulation, and employing recommended farming practices ensure high-quality coffee production."
        },
        {
            "name": "red_spider_mite",
            "description": "Red spider mites are tiny pests that feed on the underside of coffee leaves, causing yellowish or bronzed discoloration, leaf drop, and reduced vigor.\nControlling them involves spraying with approved miticides, encouraging natural predators, and using proper irrigation to maintain humidity that reduces mite populations.\nGood sanitation, weed control, and avoiding drought stress also help keep mite infestations in check."
        },
        {
            "name": "rust",
            "description": "Coffee rust, caused by the fungus Hemileia vastatrix, appears as yellow-orange powdery spots on the underside of leaves and leads to premature leaf drop and yield reduction.\nManagement includes planting resistant coffee varieties, improving farm sanitation, pruning shade trees for better airflow, and applying recommended fungicides.\nMaintaining balanced plant nutrition also improves resistance to the disease."
        }
    ],
    "corn": [
        {
            "name": "common_rust",
            "description": "Common rust in corn is caused by the fungus Puccinia sorghi, which forms reddish-brown pustules on leaves.\nSevere infections can reduce photosynthesis and yield.\nManagement relies on planting resistant hybrids, applying fungicides when warranted, and ensuring proper field sanitation.\nCrop rotation and avoiding overly dense plantings also reduce conditions conducive to rust development."
        },
        {
            "name": "gray_leaf_spot",
            "description": "Gray leaf spot is a fungal disease caused by Cercospora zeae-maydis.\nIt produces rectangular, grayish-brown lesions on leaves, interfering with photosynthesis and lowering yields.\nControl strategies include planting resistant hybrids, implementing crop rotation, and applying fungicides at the early stages of infection.\nGood residue management (tilling under infected crop debris) helps prevent overwintering of the pathogen."
        },
        {
            "name": "healthy",
            "description": "A healthy corn plant exhibits vigorous growth, strong stalks, and broad, fully extended leaves free from lesions.\nEnsuring balanced nutrition (especially nitrogen), adequate water supply, and proper planting density fosters robust growth.\nRegular scouting for early pest or disease signs and employing integrated pest management helps maintain crop health and maximize yields."
        },
        {
            "name": "northern_leaf_blight",
            "description": "Northern leaf blight is caused by the fungus Exserohilum turcicum, creating long, elliptical gray-green lesions on leaves.\nSevere cases diminish photosynthesis and yield.\nControlling northern leaf blight involves choosing resistant hybrids, applying fungicides at critical growth stages, and practicing crop rotation.\nProper residue management and balanced fertilization support plant resilience."
        }
    ],
    "cucumber": [
        {
            "name": "diseased",
            "description": "“Diseased” cucumber plants can be affected by a range of pathogens such as downy mildew, powdery mildew, bacterial wilt, or various viruses.\nSymptoms might include spotted, yellowing leaves, wilting vines, or malformed fruit.\nTreatment and prevention depend on the specific disease, but general measures include using disease-resistant varieties, ensuring adequate spacing, controlling insect vectors, and applying fungicides or bactericides as needed.\nGood sanitation and crop rotation are also key strategies."
        },
        {
            "name": "healthy",
            "description": "A healthy cucumber plant has vigorous vines, broad green leaves, and straight, uniform fruits.\nSupporting cucumber health involves using quality seeds, providing sufficient water and nutrients, and training vines for good airflow.\nRegular weeding, scouting for pests, and timely intervention with safe pesticides or biological controls keeps the crop thriving and productive."
        }
    ],
    "grape": [
        {
            "name": "black_measles (Esca or Measles)",
            "description": "Grape black measles (esca) is a complex disease caused by a group of fungi, resulting in foliar “tiger stripes,” berry spotting, and eventual vine decline.\nIt often arises from older pruning wounds.\nManaging esca involves pruning out infected wood, applying wound protectants, practicing good vineyard sanitation, and avoiding water stress.\nWhile no single cure exists, integrated vineyard management and minimizing stressors can slow disease progression."
        },
        {
            "name": "black_rot",
            "description": "Black rot in grapes, caused by Guignardia bidwellii, forms brownish-black lesions on leaves, shoots, and fruit.\nInfected grapes shrivel into hard, black mummies.\nPrevention includes removing mummified berries, pruning for better airflow, and applying protective fungicides at critical times.\nPlanting resistant cultivars and maintaining a clean vineyard floor help significantly reduce the disease’s impact."
        },
        {
            "name": "healthy",
            "description": "Healthy grapevines display lush green leaves, sturdy canes, and clusters of plump, well-formed grapes free from blemishes.\nProper canopy management, balanced fertilization, and adequate irrigation are vital for maintaining plant health.\nRegular monitoring for pests or diseases, along with timely fungicide or insecticide applications as needed, ensures a productive and high-quality harvest."
        },
        {
            "name": "leaf_blight (isariopsis_leaf_spot)",
            "description": "Isariopsis leaf spot (commonly called grape leaf blight) is a fungal disease causing dark brown spots on grape leaves.\nSevere infections may reduce photosynthetic area and ultimately lower yields.\nManagement includes applying recommended fungicides, pruning vines to improve airflow, and removing infected leaves from the vineyard.\nCrop rotation and vineyard sanitation practices also help in prevention."
        }
    ],
    "guava": [
        {
            "name": "diseased",
            "description": "Diseased guava plants may suffer from issues like anthracnose, guava wilt, or bacterial spots.\nSymptoms vary but can include leaf spots, fruit rotting, and branch dieback.\nManaging guava diseases often involves sanitation (removing infected fruit and branches), applying appropriate fungicides or bactericides, and ensuring proper nutrient and water management.\nResistant cultivars and regular orchard monitoring can greatly reduce disease losses."
        },
        {
            "name": "healthy",
            "description": "Healthy guava trees have glossy green leaves, well-formed fruits, and steady growth without visible discolorations or deformities.\nMaintaining plant health involves balanced fertilization, proper irrigation, and good orchard hygiene.\nRegular pruning to improve air circulation and sunlight penetration, along with early detection and control of pests, supports ongoing productivity."
        }
    ],
    "jamun": [
        {
            "name": "diseased",
            "description": "Jamun (Java plum) can be affected by fungal diseases like anthracnose or wilt, leading to leaf spots, fruit lesions, and branch dieback.\nProper disease management includes pruning out infected material, ensuring good soil drainage, applying recommended fungicides, and maintaining orchard cleanliness.\nChoosing disease-tolerant varieties and monitoring regularly for early symptoms helps prevent severe outbreaks."
        },
        {
            "name": "healthy",
            "description": "A healthy jamun tree exhibits lush green foliage, strong branches, and large, firm, dark-purple fruits without blemishes.\nTo keep trees healthy, provide balanced fertilization, ensure adequate watering, and prune to maintain good structure and airflow.\nRegularly inspecting trees for pests and diseases and taking prompt corrective measures supports consistent yields and good fruit quality."
        }
    ],
    "lemon": [
        {
            "name": "diseased",
            "description": "Diseased lemon trees may be affected by fungal infections (like greasy spot), bacterial infections (canker), or viral diseases.\nSymptoms range from leaf lesions and fruit blemishes to reduced vitality.\nTreatment and prevention depend on the specific pathogen, but generally involve pruning out infected parts, applying appropriate fungicides or bactericides, ensuring good orchard sanitation, and controlling insect vectors where applicable."
        },
        {
            "name": "healthy",
            "description": "A healthy lemon tree bears vibrant green leaves, fragrant blossoms, and well-shaped, brightly colored lemons.\nProper fertilization, consistent irrigation, and regular pruning encourage strong growth.\nMonitoring the tree for early pest or disease signs, employing integrated pest management strategies, and maintaining clean orchard conditions help sustain the tree’s vigor and yield."
        }
    ],
    "mango": [
        {
            "name": "diseased",
            "description": "Mango trees may suffer from maladies like anthracnose, powdery mildew, or bacterial black spot.\nThese diseases cause leaf spots, floral blight, and premature fruit drop.\nManaging mango diseases involves pruning diseased plant parts, applying recommended fungicides at key stages (like flowering), ensuring proper orchard sanitation, and selecting disease-tolerant cultivars.\nGood airflow and balanced nutrition support resilience."
        },
        {
            "name": "healthy",
            "description": "A healthy mango tree sports dark green leaves, abundant blossoms, and smooth, unblemished fruit.\nRegular fertilization, proper irrigation, and pruning for good canopy structure maintain plant vigor.\nMonitoring for early signs of disease or pests and applying preventative treatments or biological controls as needed ensure steady growth and productive harvests."
        }
    ],
    "peach": [
        {
            "name": "bacterial_spot",
            "description": "Peach bacterial spot, caused by Xanthomonas arboricola pv. pruni, creates small, dark lesions on leaves and fruits, leading to defoliation and blemished fruit.\nManagement includes planting resistant cultivars, applying copper-based bactericides, and practicing good orchard sanitation by removing infected fruit and pruned material.\nAvoiding overhead irrigation and maintaining balanced fertility also helps limit disease spread."
        },
        {
            "name": "healthy",
            "description": "A healthy peach tree has glossy leaves, robust branches, and well-formed, smooth-skinned fruit without dark lesions.\nProviding adequate nutrients, watering carefully, and pruning to improve sunlight penetration supports strong, disease-resistant growth.\nMonitoring for pests or disease and applying protective sprays when necessary keep the tree productive and the fruit quality high."
        }
    ],
    "pepper (Bell Pepper)": [
        {
            "name": "bacterial_spot",
            "description": "Bell pepper bacterial spot, caused by Xanthomonas spp., results in brown to black leaf spots and fruit blemishes, causing defoliation and unmarketable produce.\nManaging bacterial spot includes planting resistant varieties, applying copper-based bactericides, and maintaining proper field sanitation.\nAvoiding overhead irrigation and rotating crops helps reduce the pathogen’s presence in the soil."
        },
        {
            "name": "healthy",
            "description": "Healthy bell pepper plants exhibit bright green foliage, sturdy stems, and smooth, unspotted peppers.\nEnsuring balanced fertilization, proper irrigation, and adequate spacing prevents disease pressure.\nRegular scouting for pests or early disease signs allows timely intervention, and integrated pest management practices maintain plant vigor and optimize yields."
        }
    ],
    "pomegranate": [
        {
            "name": "diseased",
            "description": "Pomegranate diseases include fungal issues like alternaria fruit rot or bacterial blights, leading to leaf spots, fruit cracking, and discoloration.\nManaging these diseases involves pruning to improve airflow, removing infected fruit and leaves, and applying appropriate fungicides or bactericides.\nGood orchard hygiene, balanced nutrition, and irrigating at the soil level (rather than overhead) help reduce disease incidence."
        },
        {
            "name": "healthy",
            "description": "A healthy pomegranate tree has lush green leaves, strong branches, and firm, bright-red fruit with no signs of blemishes.\nProper fertilization, careful irrigation, and pruning to maintain an open canopy create conditions that discourage pathogens.\nRegular monitoring for insects and early diseases ensures any issues can be addressed promptly, preserving tree health and fruit quality."
        }
    ],
    "potato": [
        {
            "name": "early_blight",
            "description": "Early blight, caused by Alternaria solani, forms concentric brown rings on leaves and sometimes tubers.\nIf unmanaged, it reduces photosynthetic capacity and yield.\nPrevention includes planting disease-free seed potatoes, rotating crops, ensuring proper fertilization, and applying fungicides when conditions favor the disease.\nRemoving infected plant debris and maintaining good soil health are also essential."
        },
        {
            "name": "healthy",
            "description": "Healthy potato plants have uniform, bright green leaves, strong stems, and tubers developing underground without blemishes.\nEnsuring well-drained soil, balanced fertilization, and proper irrigation supports robust growth.\nRegular scouting for pests and disease symptoms, crop rotation, and the use of certified seed help maintain plant health and high-quality yields."
        },
        {
            "name": "late_blight",
            "description": "Late blight, caused by Phytophthora infestans, is notorious for causing dark, water-soaked lesions on leaves and stems, and rotting tubers.\nTo control late blight, plant resistant varieties, use certified seed, and apply fungicides preventatively during favorable conditions.\nRemoving infected debris, practicing crop rotation, and avoiding overhead irrigation all help reduce disease severity."
        }
    ],
    "rice": [
        {
            "name": "brown_spot",
            "description": "Brown spot in rice, caused by Bipolaris oryzae, produces brown lesions on leaves that reduce photosynthesis and weaken plants.\nManaging brown spot includes improving soil fertility (especially silica), ensuring proper irrigation, and using resistant varieties.\nFungicide applications may help in severe cases, and reducing plant stress with good agronomic practices is key to prevention."
        },
        {
            "name": "healthy",
            "description": "A healthy rice crop exhibits uniformly green leaves, steady growth, and fully developed grains.\nProviding balanced fertilization, maintaining appropriate water depth, and timely weeding support plant vigor.\nMonitoring for pests or early disease signs, rotating crops, and using resistant varieties ensure sustained yield and grain quality."
        },
        {
            "name": "hispa",
            "description": "Rice hispa is an insect pest (Dicladispa armigera) that scrapes chlorophyll from leaves, causing whitish streaks and reducing photosynthesis.\nControl methods involve applying recommended insecticides or using biological control agents (e.g., parasitoids) and maintaining well-managed fields free of weeds.\nResistant cultivars and timely insect scouting are also effective measures."
        },
        {
            "name": "leaf_blast",
            "description": "Rice leaf blast, caused by Magnaporthe oryzae, is a fungal disease that creates diamond-shaped lesions on leaves and can severely reduce yields.\nManaging blast involves planting resistant varieties, ensuring balanced fertilization (especially nitrogen), and applying fungicides when weather favors infection.\nGood field sanitation, proper irrigation management, and crop rotation further help limit outbreaks."
        },
        {
            "name": "neck_blast",
            "description": "Neck blast is a severe form of rice blast that affects the neck of the panicle, preventing grain filling and drastically cutting yields.\nSimilar management strategies apply: use resistant varieties, apply fungicides at critical times, balance nitrogen, and maintain proper field conditions.\nPrompt removal of crop residues and careful water management also reduce disease pressure."
        }
    ],
    "soybean": [
        {
            "name": "bacterial_blight",
            "description": "Bacterial blight of soybeans, caused by Pseudomonas savastanoi pv. glycinea, causes small, angular, water-soaked lesions on leaves.\nInfection can reduce yield and seed quality.\nManagement includes using disease-free seed, rotating crops, and applying copper-based bactericides.\nResistant cultivars and good field sanitation also help prevent spread."
        },
        {
            "name": "caterpillar",
            "description": "Various caterpillars (e.g., soybean looper or armyworms) feed on soybean leaves and pods, reducing yield and quality.\nControlling caterpillars involves scouting to determine infestation levels, applying insecticides if thresholds are met, and encouraging natural predators like parasitic wasps.\nProper field management, crop rotation, and timing of planting also help reduce caterpillar pressure."
        },
        {
            "name": "diabrotica_speciosa",
            "description": "Diabrotica speciosa (a cucumber beetle species) is a pest that can feed on soybean leaves, roots, and pods.\nIt may cause stunted growth and reduced yield.\nManagement strategies include rotating crops to disrupt life cycles, using insecticides if necessary, and employing biological controls like entomopathogenic fungi.\nGood field sanitation and resistant cultivars can further minimize damage."
        },
        {
            "name": "downy_mildew",
            "description": "Downy mildew on soybeans, caused by Peronospora manshurica, produces pale green to yellow spots on leaves with a downy growth on the underside.\nThough rarely severe, it can reduce seed quality.\nControl measures include using resistant varieties, applying fungicides, and maintaining good air circulation.\nRemoving infected plant debris and rotating with non-host crops also help."
        },
        {
            "name": "healthy",
            "description": "Healthy soybean plants have uniform stands, robust leaf canopies, and well-filled pods without blemishes.\nEnsuring balanced fertilization, proper irrigation, and timely weed control fosters good growth.\nRegular scouting for pests and early disease intervention, along with crop rotation, maintains crop health and optimizes yield."
        },
        {
            "name": "mosaic_virus",
            "description": "Soybean mosaic virus causes mosaic patterns, distortion, and stunting in infected plants, lowering yield and seed quality.\nControl focuses on using virus-free seed, planting resistant varieties, and managing aphid populations that transmit the virus.\nPrompt removal of infected plants and maintaining field hygiene are also effective preventive steps."
        },
        {
            "name": "powdery_mildew",
            "description": "Powdery mildew in soybeans, caused by Microsphaera diffusa, forms white, powdery patches on leaves, potentially reducing photosynthetic activity.\nManaging powdery mildew includes planting resistant varieties, ensuring proper plant spacing, and applying fungicides if severe.\nGood air circulation and crop rotation help prevent significant outbreaks."
        },
        {
            "name": "rust",
            "description": "Soybean rust, caused by Phakopsora pachyrhizi, forms small, reddish-brown pustules on leaves.\nThis disease can rapidly defoliate plants and severely impact yields.\nControl strategies include using resistant varieties, applying fungicides at early detection, and rotating with non-host crops.\nMonitoring weather conditions that favor rust spread and timely scouting are crucial."
        },
        {
            "name": "southern_blight",
            "description": "Southern blight, caused by Sclerotium rolfsii, affects stems near the soil line, causing wilt and plant death.\nControlling southern blight includes crop rotation with non-host species, deep plowing to bury fungal sclerotia, and applying fungicides when necessary.\nMaintaining clean fields and avoiding over-irrigation also help reduce disease incidence."
        }
    ],
    "strawberry": [
        {
            "name": "leaf_scorch",
            "description": "Leaf scorch on strawberries, often caused by Diplocarpon earlianum, appears as irregular, dark purple to brown spots on leaves, leading to a scorched appearance.\nManaging leaf scorch involves selecting resistant varieties, removing infected leaves, applying fungicides during vulnerable growth stages, and improving air circulation with proper plant spacing.\nGood sanitation and crop rotation also help limit the disease."
        },
        {
            "name": "healthy",
            "description": "A healthy strawberry plant has vibrant green leaves, strong runners, and bright red, fully formed berries without spots or deformities.\nEnsuring adequate fertilization, proper watering, and well-drained soil supports plant health.\nRegularly removing weeds and monitoring for pests or early disease symptoms helps maintain a productive and disease-free strawberry patch."
        }
    ],
    "sugarcane": [
        {
            "name": "bacterial_blight",
            "description": "Sugarcane bacterial blight, caused by Xanthomonas albilineans, leads to leaf chlorosis, white leaf stripes, and stunted cane growth.\nControl measures include using clean planting material, rotating crops, and selecting resistant varieties.\nMaintaining field sanitation and applying recommended bactericides (if available) can help minimize yield losses."
        },
        {
            "name": "healthy",
            "description": "A healthy sugarcane crop shows tall, thick canes with lush green leaves free from streaks or lesions.\nEnsuring adequate nutrient supply, regular irrigation, and proper weed management fosters robust growth.\nRoutine scouting for pests and diseases, coupled with integrated pest management and good field hygiene, maintains stand health and productivity."
        },
        {
            "name": "red_rot",
            "description": "Red rot, caused by Colletotrichum falcatum, is a devastating fungal disease that reddens cane internodes, leading to a foul smell, hollow canes, and reduced sugar content.\nManagement focuses on planting resistant varieties, removing infected canes, and maintaining clean fields.\nFungicidal treatments, proper irrigation, and avoiding waterlogging also help limit the disease."
        },
        {
            "name": "red_stripe",
            "description": "Red stripe (or top rot) is caused by Acidovorax avenae subsp. avenae, creating red stripes on leaves and eventual top rot.\nUsing disease-free seed sets, practicing crop rotation, and improving drainage can help reduce its incidence.\nChemical control options are limited, so preventing spread through sanitation and choosing tolerant varieties is crucial."
        },
        {
            "name": "rust",
            "description": "Sugarcane rust, caused by Puccinia melanocephala, produces reddish-brown pustules on leaves, reducing photosynthesis and yield.\nControlling rust involves planting resistant varieties, timing harvest to avoid peak infection, and using fungicides as recommended.\nEnsuring balanced nutrition and proper field sanitation also contributes to disease management."
        }
    ],
    "tea": [
        {
            "name": "algal_leaf (Red Rust)",
            "description": "Algal leaf spot in tea, often called red rust, is caused by the alga Cephaleuros virescens.\nIt forms reddish-brown spots on leaves and can weaken the bush.\nManagement includes pruning to improve sunlight and airflow, removing infected leaves, and applying copper-based fungicides.\nMaintaining balanced nutrition and good field sanitation helps prevent outbreaks."
        },
        {
            "name": "anthracnose",
            "description": "Anthracnose in tea, caused by fungi like Colletotrichum spp., creates brown, sunken lesions on leaves and shoots.\nIt can reduce yield and affect leaf quality.\nControl strategies involve pruning diseased parts, applying approved fungicides, and ensuring proper shade management.\nKeeping tea bushes healthy with balanced fertilizer and proper irrigation also helps prevent anthracnose."
        },
        {
            "name": "bird_eye_spot",
            "description": "Bird’s eye spot, caused by Cercospora theae, results in small, round spots with a distinctive halo on tea leaves, resembling a bird’s eye.\nIt can lower tea quality by damaging leaves.\nControlling this disease involves applying suitable fungicides, pruning to reduce humidity within the canopy, and ensuring good field sanitation.\nResistant varieties and balanced nutrition further help manage this issue."
        },
        {
            "name": "brown_blight",
            "description": "Brown blight in tea, often caused by Glomerella cingulata, leads to browning and leaf fall, reducing the number of harvestable leaves.\nManagement includes pruning affected areas, ensuring good air circulation, and applying fungicides when necessary.\nProper shade regulation, balanced fertilizer application, and weed control enhance plant resilience."
        },
        {
            "name": "healthy",
            "description": "Healthy tea bushes have uniformly green leaves, steady flush production, and minimal leaf blemishes.\nMaintaining proper soil pH, adequate fertilizer, consistent moisture, and suitable shade levels supports leaf quality and yield.\nRegular monitoring for pests and diseases and prompt interventions sustain healthy, high-quality tea crops."
        }
    ],
    "tomato": [
        {
            "name": "Bacterial_spot",
            "description": "Tomato bacterial spot, caused by Xanthomonas spp., creates dark, water-soaked spots on leaves and fruit, leading to defoliation and blemished tomatoes.\nControl measures include planting disease-free seed, applying copper-based bactericides, and avoiding overhead irrigation.\nCrop rotation, good sanitation, and choosing resistant cultivars help minimize bacterial spot."
        },
        {
            "name": "Early_blight",
            "description": "Early blight, caused by Alternaria solani, forms concentric brown rings on older leaves and can spread to stems and fruit.\nPreventing early blight involves proper crop rotation, applying fungicides when conditions favor disease, and removing infected debris.\nEnsuring balanced nutrition and adequate spacing improves airflow and reduces disease severity."
        },
        {
            "name": "Late_blight",
            "description": "Late blight, caused by Phytophthora infestans, creates water-soaked, dark lesions on leaves, stems, and fruit.\nIt can rapidly devastate a tomato crop.\nControl focuses on planting resistant varieties, applying fungicides preventively, and avoiding overhead irrigation.\nPrompt removal of infected plants and rotating with non-host crops also limit disease spread."
        },
        {
            "name": "Leaf_Mold",
            "description": "Leaf mold in tomatoes, caused by Passalora fulva, produces yellowish spots on leaf tops and fuzzy brown or purple mold underneath.\nControlling leaf mold includes ensuring good ventilation (staking or trellising plants), applying fungicides, and removing infected leaves.\nAvoiding high humidity and dense planting further helps in prevention."
        },
        {
            "name": "Septoria_leaf_spot",
            "description": "Septoria leaf spot, caused by Septoria lycopersici, appears as numerous small, dark spots with light centers on lower leaves.\nIt defoliates plants and lowers yields.\nManagement includes using disease-free seed, rotating crops, applying fungicides, and removing infected leaves.\nMaintaining good airflow and avoiding overhead watering are also essential."
        },
        {
            "name": "Spider_mites Two-spotted_spider_mite",
            "description": "Two-spotted spider mites cause stippling, yellowing, and webbing on tomato leaves, reducing plant vigor.\nControl involves using miticides or insecticidal soaps, encouraging natural predators (such as predatory mites), and maintaining adequate humidity to discourage mite outbreaks.\nEarly detection and removing heavily infested leaves help limit damage."
        },
        {
            "name": "Target_Spot",
            "description": "Target spot, caused by Corynespora cassiicola, forms small, dark lesions with lighter centers on leaves and can also affect fruit.\nManaging target spot includes rotating crops, removing infected debris, applying fungicides, and ensuring good ventilation.\nStaking plants and maintaining proper plant spacing reduce leaf wetness and slow disease spread."
        },
        {
            "name": "Tomato_Yellow_Leaf_Curl_Virus",
            "description": "TYLCV is transmitted by whiteflies and causes yellowing and curling of leaves, stunting, and reduced yields.\nControl focuses on managing whiteflies with insecticides, reflective mulches, or biological controls.\nUsing resistant varieties, removing infected plants, and practicing good field hygiene help contain virus spread."
        },
        {
            "name": "Tomato_mosaic_virus",
            "description": "Tomato mosaic virus (ToMV) causes mottled, distorted leaves and reduced fruit quality.\nIt spreads mechanically via contaminated tools and hands.\nPreventing ToMV includes using virus-free seed, disinfecting tools, removing infected plants, and practicing crop rotation.\nResistant varieties and strict sanitation measures are key to control."
        },
        {
            "name": "healthy",
            "description": "A healthy tomato plant has firm, green foliage, sturdy stems, and bright, smooth fruits free of spots or deformities.\nEnsuring balanced nutrition, proper irrigation, and adequate spacing allows good airflow and reduces disease risk.\nRegular scouting, integrated pest management, and timely maintenance result in vigorous plants and abundant harvests."
        }
    ]
}

found_diseases = []

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        customtkinter.set_appearance_mode("System")
        customtkinter.set_default_color_theme("blue")

        self.title("Sci Research 1.0")
        self.geometry("1200x800")  # Adjusted size for better visibility
        self.minsize(800, 600)  # Set a minimum size

        # Configure grid layout for the main window
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Sidebar Frame
        self.sidebar_frame = customtkinter.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure((0, 1, 2, 3, 4, 5, 6), weight=0)
        self.sidebar_frame.grid_rowconfigure(7, weight=1)  # For radio buttons at the bottom

        # Logo Label
        self.logo_label = customtkinter.CTkLabel(
            self.sidebar_frame,
            text="SciResearch v1",
            font=customtkinter.CTkFont(size=20, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Run Button
        self.sidebar_button_run = customtkinter.CTkButton(
            self.sidebar_frame,
            command=self.run_button_event,
            text='Run',
            width=160
        )
        self.sidebar_button_run.grid(row=1, column=0, padx=20, pady=(10, 5))

        # Stop Button
        self.sidebar_button_stop = customtkinter.CTkButton(
            self.sidebar_frame,
            command=self.stop_button_event,
            text='Stop',
            state='disabled',
            width=160
        )
        self.sidebar_button_stop.grid(row=2, column=0, padx=20, pady=5)

        # File Selection Button
        self.file_button = customtkinter.CTkButton(
            self.sidebar_frame,
            text="Select File",
            command=self.select_file,
            width=160
        )
        self.file_button.grid(row=3, column=0, padx=20, pady=5)

        # Radio Buttons for Detection Mode
        self.radio_var = tk.IntVar(value=0)
        self.radio_button_live = customtkinter.CTkRadioButton(
            self.sidebar_frame,
            text="Live Detection",
            variable=self.radio_var,
            value=0
        )
        self.radio_button_live.grid(row=4, column=0, padx=20, pady=(20, 5), sticky="w")

        self.radio_button_file = customtkinter.CTkRadioButton(
            self.sidebar_frame,
            text="File Detection",
            variable=self.radio_var,
            value=1
        )
        self.radio_button_file.grid(row=5, column=0, padx=20, pady=5, sticky="w")

        # Spacer to push radio buttons to the bottom
        self.sidebar_frame.grid_rowconfigure(6, weight=1)

        # Main Content Frame
        self.main_frame = customtkinter.CTkFrame(self, corner_radius=0)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Video Display
        self.video_label = customtkinter.CTkLabel(self.main_frame, text='', anchor="center")
        self.video_label.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Lower Frame for Response
        

        

        # Tab View
        self.tabview = customtkinter.CTkTabview(self.main_frame, width=350, height=800)
        self.tabview.grid(row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nsew")
        self.tabview.add("Output")
        self.tabview.add("Settings")
        self.tabview.grid_rowconfigure(0, weight=1)
        self.tabview.grid_columnconfigure(0, weight=1)

        # Plant Selection Dropdown in Output Tab
        self.plant_dropdown = customtkinter.CTkOptionMenu(
            self.tabview.tab('Output'),
            dynamic_resizing=True,
            values=classes,
            command=self.model_select
        )
        self.plant_dropdown.set("Select Plant")
        self.plant_dropdown.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")

        # Disease Information Display
        self.disease_info_label = customtkinter.CTkLabel(
            self.tabview.tab('Output'),
            text="Disease info and treatment:",
            anchor="w",
            justify="left",
            font=customtkinter.CTkFont(weight="bold")
        )
        self.disease_info_label.grid(row=1, column=0, padx=20, pady=(10, 5), sticky="w")

        self.disease_info_text = customtkinter.CTkTextbox(
            self.tabview.tab('Output'),
            height=800,
            width=400,
            wrap="word"
        )
        self.disease_info_text.grid(row=2, column=0, padx=20, pady=(0, 20), sticky="nsew")
        self.disease_info_text.configure(state='disabled')  # Initially disabled

        # Settings Tab - Confidence Slider
        self.confidence_slider = customtkinter.CTkSlider(
            self.tabview.tab('Settings'),
            from_=0,
            to=1,
            number_of_steps=100,
            command=self.confidence_function
        )
        self.confidence_slider.set(confidence)
        self.confidence_slider.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="ew")
        self.confidence_label = customtkinter.CTkLabel(
            self.tabview.tab('Settings'),
            text=f'Confidence > {confidence * 100:.1f}%',
            anchor="w"
        )
        self.confidence_label.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="w")
        
        self.zoom_slider = customtkinter.CTkSlider(
            self.tabview.tab('Settings'),
            from_=0,
            to=1,
            number_of_steps=100,
            command=self.zoom_function
        )
        self.zoom_slider.set(scale_factor)
        self.zoom_slider.grid(row=1, column=0, padx=20, pady=(20, 10), sticky="ew")
        self.zoom_label = customtkinter.CTkLabel(
            self.tabview.tab('Settings'),
            text='zoom',
            anchor="w"
        )
        self.zoom_label.grid(row=2, column=0, padx=20, pady=(20, 10), sticky="ew")

        # Initialize variables
        self.plant_type = 'apple'
        self.file_path = None

    def get_selected_value(self):
        self.plant_type = self.plant_dropdown.get()
        print(f"Selected Plant: {self.plant_type}")
        self.model_select(self.plant_type)

    def select_file(self):
        self.file_path = filedialog.askopenfilename(
            title="Select a Video File",
            filetypes=(("MP4 files", "*.mp4"), ("All files", "*.*"))
        )
        if self.file_path:
            print(f"Selected file: {self.file_path}")

    def run_button_event(self):
        global stopped, scale_factor
        scale_factor = 0.4 
        stopped = False
        self.sidebar_button_run.configure(state="disabled", text="Running...")
        self.sidebar_button_stop.configure(state='enabled', text='Stop')
        if self.radio_var.get() == 0:
            threading.Thread(target=self.live_detection_thread, daemon=True).start()
        else:
            threading.Thread(target=self.file_detection_thread, daemon=True).start()

    def stop_button_event(self):
        global stopped
        stopped = True
        self.sidebar_button_run.configure(state="enabled", text="Run")
        self.sidebar_button_stop.configure(state='disabled', text='Stop')     

    def model_select(self, crop):
        global leaf_model, class_name, input_details, output_details

        # Update self.plant_type based on the selected dropdown value
        self.plant_type = crop

        # Load TFLite model and allocate tensors
        leaf_model = tf.lite.Interpreter(model_path=f'/Users/joshua.stanley/Desktop/Science Research/Saved Models/tflite2/model{crop}.tflite')
        leaf_model.allocate_tensors()

        # Get input and output details
        input_details = leaf_model.get_input_details()
        output_details = leaf_model.get_output_details()

        # Print input/output details for debugging
        print("pInput Details:", input_details)
        print("Output Details:", output_details)

        class_name = globals()[f'{crop}_classes']
        print(f"Selected Crop: {crop}")
        print(f"Class Names: {class_name}")
        print(f"Plant type updated to: {self.plant_type}")

    def confidence_function(self, value):
        global confidence
        confidence = float(value)
        self.confidence_label.configure(text=f'Confidence > {confidence * 100:.1f}%')
        print(f'Confidence > {confidence * 100:.1f}%')
    def zoom_function(self, value):
        global scale_factor
        scale_factor = float(value)
        
    def update_disease_info(self, info):
        self.disease_info_text.configure(state='normal')
        self.disease_info_text.delete("1.0", tk.END)
        self.disease_info_text.insert(tk.END, info)

        self.disease_info_text.configure(state='disabled')

    def live_detection_thread(self):
        model = YOLO('/Users/joshua.stanley/Desktop/train32/weights/best.pt')
        model.overrides['verbose'] = False
        cap = cv2.VideoCapture(0)
        cap.set(3, 1000)
        cap.set(4, 500)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame for zoom-out effect (scale factor < 1)
              # Adjust this factor as needed (e.g., 0.5 for 50% size)
            frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

            results = model(frame)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    if stopped:
                        break
                    if conf >= confidence:
                        cropped_leaf = frame[y1:y2, x1:x2]
                        if cropped_leaf.size == 0:
                            print("Cropped leaf is empty! Check bounding box coordinates.")
                        else:
                            print("Cropped leaf shape:", cropped_leaf.shape)

                        cropped_leaf = cv2.cvtColor(cropped_leaf, cv2.COLOR_BGR2RGB)
                        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')  # Unique timestamp
                        save_path = os.path.join(output_dir, f"leaf_{timestamp}.jpg")
                        cv2.imwrite(save_path, cv2.cvtColor(cropped_leaf, cv2.COLOR_RGB2BGR))
                        print(f"Saved cropped leaf to {save_path}")
                        data = tf.image.resize(cropped_leaf, [128, 128])
                        data = np.expand_dims(data, axis=0)  # Shape [1, 128, 128, 3], assuming model expects this

                        # Set the input tensor
                        leaf_model.set_tensor(input_details[0]['index'], data)

                        # Invoke the model
                        leaf_model.invoke()

                        # Get the prediction output
                        prediction = leaf_model.get_tensor(output_details[0]['index'])  # This will give you the raw output array

                        # Fixed preprocessing pipeline
                        confidence_threshold = 0.4  # Set your desired confidence threshold

                        predicted_class = np.argmax(prediction[0])
                        conf_model = prediction[0][predicted_class]  # Get confidence for the predicted class

                        if conf_model < confidence_threshold:  # Check if confidence is below the threshold
                            class_label = "get closer to leaf"
                            color = (255, 255, 0)  # Yellow for low confidence
                            label = class_label
                        else:
                            class_label = class_name[predicted_class]
                            color = (0, 255, 0) if 'healthy' in class_label else (0, 0, 255)
                            label = f"{class_label} ({conf_model:.2f})"

                        print(class_label)

                        # Draw rectangle and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb_frame))
            self.video_label.configure(image=img)
            self.video_label.image = img

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.sidebar_button_run.configure(state="enabled", text="Run")

    def file_detection_thread(self):
        if not self.file_path:
            print("No file selected for detection.")
            self.sidebar_button_run.configure(state="enabled", text="Run")
            return

        model = YOLO('/Users/joshua.stanley/Desktop/train32/weights/best.pt')
        model.overrides['verbose'] = False
        cap = cv2.VideoCapture(self.file_path)
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            
            frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

            results = model(frame)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0]
                    if stopped:
                        cap.release()
                        cv2.destroyAllWindows()
                        break
                    if conf >= confidence:
                        cropped_leaf = frame[y1:y2, x1:x2]
                        if cropped_leaf.size == 0:
                            print("Cropped leaf is empty! Check bounding box coordinates.")
                        else:
                            print("Cropped leaf shape:", cropped_leaf.shape)
                        cropped_leaf = cv2.cvtColor(cropped_leaf, cv2.COLOR_BGR2RGB)
                        data = tf.image.resize(cropped_leaf, [128, 128])
                        data = np.expand_dims(data, axis=0)  # Shape [1, 128, 128, 3], assuming model expects this

                        # Set the input tensor
                        leaf_model.set_tensor(input_details[0]['index'], data)

                        # Invoke the model
                        leaf_model.invoke()

                        # Get the prediction output
                        prediction = leaf_model.get_tensor(output_details[0]['index'])  # This will give you the raw output array

                        # Fixed preprocessing pipeline
                        confidence_threshold = 0.4  # Set your desired confidence threshold

                        predicted_class = np.argmax(prediction[0])
                        conf_model = prediction[0][predicted_class]  # Get confidence for the predicted class

                        if conf_model < confidence_threshold:  # Check if confidence is below the threshold
                            class_label = "get closer to leaf"
                            color = (255, 255, 0)  # Yellow for low confidence
                            label = class_label
                        else:
                            class_label = class_name[predicted_class]
                            color = (0, 255, 0) if 'healthy' in class_label else (0, 0, 255)
                            label = f"{class_label} ({conf_model:.2f})"

                        print(class_label)

                        try:
                            disease_info = disease_dictionary[self.plant_type][predicted_class]
                            if disease_info['description'] not in found_diseases:
                                found_diseases.append(disease_info['description'])
                                found_diseases.append('\n')
                                self.update_disease_info(found_diseases)
                        except:
                            pass

                        # Draw rectangle and label
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(rgb_frame))
            self.video_label.configure(image=img)
            self.video_label.image = img
        cap.release()
        cv2.destroyAllWindows()
        self.sidebar_button_run.configure(state="enabled", text="Run")    

if __name__ == "__main__":
    app = App()
    app.mainloop()