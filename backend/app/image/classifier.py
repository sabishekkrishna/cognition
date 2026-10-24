import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
import math
import os
from typing import List, Dict, Any, Optional, Tuple
from ..schemas.schemas import ModelPredictions, PredictionItem

# Standard ImageNet Class Labels (1000 categories subset/compact mapping)
# Torchvision default class names
IMAGENET_CATEGORIES = [
    "tench", "goldfish", "great white shark", "tiger shark", "hammerhead", "electric ray", "stingray", "cock",
    "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "robin", "bulbul",
    "jay", "magpie", "chickadee", "water ouzel", "kite", "bald eagle", "vulture", "great grey owl", "black grouse",
    "ptarmigan", "ruffed grouse", "prairie chicken", "peacock", "quail", "partridge", "african grey", "macaw",
    "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan",
    "drake", "red-breasted merganser", "goose", "black swan", "white stork", "black stork", "spoonbill", "flamingo",
    "little blue heron", "american egret", "bittern", "crane", "limpkin", "european gallinule", "american coot",
    "bustard", "ruddy turnstone", "red-backed sandpiper", "redshank", "dowitcher", "oystercatcher", "pelican",
    "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "chihuahua", "japanese spaniel",
    "maltese dog", "pekinese", "shih-tzu", "blenheim spaniel", "papillon", "toy terrier", "rhodesian ridgeback",
    "afghan hound", "basset", "beagle", "bloodhound", "bluetick", "black-and-tan coonhound", "walker hound",
    "english foxhound", "redbone", "borzoi", "irish wolfhound", "italian greyhound", "whippet", "ibizan hound",
    "norwegian elkhound", "otterhound", "saluki", "scottish deerhound", "weimaraner", "staffordshire bullterrier",
    "american staffordshire terrier", "bedlington terrier", "border terrier", "kerry blue terrier", "irish terrier",
    "norfolk terrier", "norwich terrier", "yorkshire terrier", "wire-haired fox terrier", "lakeland terrier",
    "sealyham terrier", "airedale", "cairn", "australian terrier", "dandie dinmont", "boston bull", "miniature schnauzer",
    "giant schnauzer", "standard schnauzer", "scotch terrier", "tibetan terrier", "silky terrier", "soft-coated wheaten terrier",
    "west highland white terrier", "lhasa", "flat-coated retriever", "curly-coated retriever", "golden retriever",
    "labrador retriever", "chesapeake bay retriever", "german short-haired pointer", "vizsla", "english setter",
    "irish setter", "gordon setter", "brittany spaniel", "clumber", "english springer", "welsh springer spaniel",
    "cocker spaniel", "sussex spaniel", "irish water spaniel", "kuvasz", "schipperke", "groenendael", "malinois",
    "briard", "kelpie", "komondor", "old english sheepdog", "shetland sheepdog", "collie", "border collie",
    "bouvier des flandres", "rottweiler", "german shepherd", "doberman", "miniature pinscher", "greater swiss mountain dog",
    "bernese mountain dog", "appenzeller", "entlebucher", "boxer", "bull mastiff", "tibetan mastiff", "french bulldog",
    "great dane", "saint bernard", "eskimo dog", "malamute", "siberian husky", "affenpinscher", "basenji", "pug",
    "leonberg", "newfoundland", "great pyrenees", "samoyed", "pomeranian", "chow", "keeshond", "brabancon griffon",
    "pembroke", "cardigan", "toy poodle", "miniature poodle", "standard poodle", "mexican hairless", "timber wolf",
    "white wolf", "red wolf", "coyote", "dingo", "dhole", "african hunting dog", "hyena", "red fox", "kit fox",
    "arctic fox", "grey fox", "tabby", "tiger cat", "persian cat", "siamese cat", "egyptian cat", "cougar", "lynx",
    "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "american black bear", "ice bear",
    "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "long-horned beetle", "leaf beetle",
    "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket", "walking stick",
    "cockroach", "mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "admiral", "ringlet",
    "monarch", "cabbage butterfly", "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin",
    "sea cucumber", "wood rabbit", "hare", "angora", "hamster", "porcupine", "fox squirrel", "marmot", "beaver",
    "guinea pig", "sorrel", "zebra", "hog", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison",
    "ram", "bighorn", "ibex", "hartebeest", "impala", "gazelle", "arabian camel", "llama", "weasel", "mink",
    "polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan",
    "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas", "baboon", "macaque", "langur", "colobus",
    "proboscis monkey", "marmoset", "capuchin", "howler monkey", "titi", "spider monkey", "squirrel monkey",
    "madagascar cat", "indri", "indian elephant", "african elephant", "lesser panda", "giant panda", "barracouta",
    "eel", "coho", "rock beauty", "anemone fish", "sturgeon", "gar", "lionfish", "puffer", "abacus", "abaya",
    "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance",
    "amphibian", "analog clock", "apiary", "apron", "ashcan", "assault rifle", "backpack", "bakery", "balance beam",
    "balloon", "ballpoint", "band aid", "banjo", "bannister", "barbell", "barber chair", "barbershop", "barn",
    "barometer", "barrel", "barrow", "baseball", "basketball", "bassinet", "bassoon", "bathing cap", "bath towel",
    "bathtub", "beach wagon", "beacon", "beaker", "bearskin", "beer bottle", "beer glass", "bell cote", "bib",
    "bicycle-built-for-two", "bikini", "binder", "binoculars", "birdhouse", "boathouse", "bobsled", "bolo tie",
    "bonnet", "bookcase", "bookshop", "bottlecap", "bow", "bow tie", "brass", "brassiere", "breakwater", "breastplate",
    "broom", "bucket", "buckle", "bulletproof vest", "bullet train", "butcher shop", "cab", "caldron", "candle",
    "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "carpenter's kit", "carton", "car wheel",
    "cash machine", "cassette", "cassette player", "castle", "catamaran", "cd player", "cello", "cellular telephone",
    "chain", "chainlink fence", "chain mail", "chain saw", "chest", "chiffonier", "chime", "china cabinet", "christmas stocking",
    "cliff dwelling", "cloak", "clog", "cocktail shaker", "coffee mug", "coffeepot", "coil", "combination lock",
    "computer keyboard", "confectionery", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot",
    "cowboy hat", "cradle", "crane", "crash helmet", "crate", "crib", "crock pot", "croquet ball", "crutch",
    "cuirass", "dam", "desk", "desktop computer", "dial telephone", "diaper", "digital clock", "digital watch",
    "dining table", "dishrag", "dishwasher", "disk brake", "dock", "dogsled", "dome", "doormat", "drilling platform",
    "drum", "drumstick", "dumbbell", "dugout", "electric fan", "electric guitar", "electric locomotive", "entertainment center",
    "envelope", "espresso maker", "face powder", "feather boa", "file", "fireboat", "fire engine", "fire screen",
    "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster",
    "freight car", "french horn", "frying pan", "fur coat", "garbage truck", "gasmask", "gas pump", "goblet", "go-kart",
    "golf ball", "golfcart", "gondola", "gong", "gown", "grand piano", "greenhouse", "grille", "grocery store",
    "guillotine", "hair slide", "hair spray", "half track", "hammer", "hamper", "hand blower", "hand-held computer",
    "handkerchief", "hard disc", "harmonica", "harp", "harvester", "hatchet", "holster", "home theater", "honeycomb",
    "hook", "hoopskirt", "horizontal bar", "horse cart", "hourglass", "ipod", "iron", "ironing board", "jack-o'-lantern",
    "jean", "jeep", "jersey", "jigsaw puzzle", "jinrikisha", "joystick", "kimono", "knee pad", "knot", "lab coat",
    "ladle", "lampshade", "laptop", "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter",
    "limousine", "liner", "lipstick", "loafer", "lotion", "loudspeaker", "loupe", "lumbermill", "magnetic compass",
    "mailbag", "mailbox", "maillot", "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze",
    "measuring cup", "medicine chest", "megalith", "microphone", "microwave", "military uniform", "milk can", "minibus",
    "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "model t", "modem", "monastery",
    "monitor", "moped", "mortar", "mortarboard", "mosque", "mosquito net", "motor scooter", "mountain bike",
    "mountain tent", "mouse", "mousetrap", "moving van", "muzzle", "nail", "neck brace", "necklace", "nipple",
    "notebook", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "organ", "oscilloscope", "overskirt",
    "oxcart", "oxygen mask", "packet", "paddle", "paddlewheel", "padlock", "paintbrush", "pajama", "palace",
    "panpipe", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "passenger car", "patio",
    "pay-phone", "pedestal", "pencil box", "pencil sharpener", "perfume", "petri dish", "photocopier", "pick",
    "pickelhaube", "picket fence", "pickup", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball",
    "pinwheel", "pirate", "pitcher", "plane", "planetarium", "plastic bag", "plate rack", "plow", "plunger",
    "polaroid camera", "pole", "police van", "poncho", "pool table", "pop bottle", "pot", "potter's wheel",
    "power drill", "prayer rug", "printer", "prison", "projectile", "projector", "puck", "punching bag", "purse",
    "quill", "quilt", "racer", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle",
    "reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair",
    "rotisserie", "rubber eraser", "rugby ball", "rule", "running shoe", "safe", "safety pin", "saltshaker", "sandal",
    "sarong", "sax", "scabbard", "scale", "school bus", "schooner", "scoreboard", "screen", "screw", "screwdriver",
    "seat belt", "sewing machine", "shield", "shoe shop", "shoji", "shopping basket", "shopping cart", "shovel",
    "shower cap", "shower curtain", "ski", "ski mask", "sleeping bag", "slide rule", "sliding door", "slot",
    "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar dish", "sombrero",
    "soup bowl", "space bar", "space heater", "space shuttle", "spatula", "speedboat", "spider web", "spindle",
    "split-rail fence", "spoke shave", "sports car", "spotlight", "stage", "steam locomotive", "steel arch bridge",
    "steel drum", "stethoscope", "stole", "stone wall", "stopwatch", "stove", "strainer", "streetcar", "stretcher",
    "studio couch", "stupa", "submarine", "suit", "sundial", "sunglass", "sunglasses", "sunscreen", "suspension bridge",
    "swab", "sweatshirt", "swimming trunks", "swing", "switch", "syringe", "table lamp", "tank", "tape player",
    "teapot", "teddy", "television", "tennis ball", "thatch", "theater curtain", "thimble", "thresher", "throne",
    "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toyshop", "tractor",
    "trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus",
    "trombone", "tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright", "vacuum", "vase",
    "vault", "velvet", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock",
    "wallet", "wardrobe", "warplane", "washbasin", "washer", "water bottle", "water jug", "water tower", "whiskey jug",
    "whistle", "wig", "window screen", "window shade", "windsor tie", "wine bottle", "wing", "wok", "wooden spoon",
    "wool", "worm fence", "wreck", "yawl", "yurt", "web site", "comic book", "crossword puzzle", "street sign",
    "traffic light", "book jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream",
    "ice lolly", "french loaf", "bagel", "pretzel", "cheeseburger", "hotdog", "mashed potato", "head cabbage",
    "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber",
    "artichoke", "bell pepper", "cardoon", "mushroom", "granny smith", "strawberry", "orange", "lemon", "fig",
    "pineapple", "banana", "jackfruit", "custard apple", "pomegranate", "hay", "carbonara", "chocolate sauce",
    "dough", "meat loaf", "pizza", "potpie", "burrito", "red wine", "espresso", "cup", "eggnog", "alp", "bubble",
    "cliff", "coral reef", "geyser", "lakeside", "promontory", "sandbar", "seashore", "valley", "volcano",
    "ballplayer", "groom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "hip",
    "buckeye", "coral fungus", "agaric", "gyromitra", "stinkhorn", "earthstar", "hen-of-the-woods", "bolete",
    "ear", "toilet tissue"
]

class MobileNetV3Classifier:
    """
    Local MobileNetV3-Small classifier.
    Exposes top-k predictions, intermediate 576-dim feature embeddings,
    entropy, and margin between top two predictions.
    """
    def __init__(self):
        self.device = torch.device("cpu")
        self.active_version = "pretrained-mobilenetv3-small"
        self.is_custom = False
        self.custom_classes = None

        # Standard ImageNet preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self._load_base_model()

    def _load_base_model(self):
        try:
            # Use torchvision pretrained weights
            weights = models.MobileNet_V3_Small_Weights.DEFAULT
            self.model = models.mobilenet_v3_small(weights=weights)
        except Exception as e:
            # Fallback if network download fails: initialize architecture
            print(f"Warning loading pretrained weights: {e}. Initializing unweighted architecture.")
            self.model = models.mobilenet_v3_small(weights=None)

        self.model.eval()
        self.model.to(self.device)
        self.active_version = "pretrained-mobilenetv3-small"
        self.is_custom = False
        self.custom_classes = None

    def revert_to_known_good(self):
        """Revert back to known-good pretrained base model."""
        self._load_base_model()
        return {"status": "reverted", "active_version": self.active_version}

    def load_custom_checkpoint(self, checkpoint_path: str, classes: List[str], version_name: str):
        """Loads a locally trained checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

        num_classes = len(classes)
        # Re-create model architecture with custom classification head
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[0].in_features
        # Replace classifier
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.Hardswish(),
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(1024, num_classes),
        )
        state_dict = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.device)

        self.model = model
        self.active_version = version_name
        self.is_custom = True
        self.custom_classes = classes

    def predict(self, image_bgr: np.ndarray, top_k: int = 5) -> ModelPredictions:
        """
        Runs local inference on a single image.
        Returns top-k predictions, intermediate embedding sample, entropy, and top-two margin.
        """
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Invalid image input for inference.")

        # Convert OpenCV BGR to RGB
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)

        # Hook or extract intermediate feature embedding (576-dim before classifier)
        with torch.no_grad():
            features = self.model.features(input_tensor)
            pooled = self.model.avgpool(features)
            embedding = torch.flatten(pooled, 1) # Shape: (1, 576)
            logits = self.model.classifier(embedding) # Shape: (1, num_classes)
            probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        embedding_vector = embedding.squeeze(0).cpu().numpy()
        embedding_sample = [float(round(v, 4)) for v in embedding_vector[:16]] # 16 sample dims for lightweight inspection

        # Compute prediction entropy: H = -sum(p * log2(p))
        # High entropy = uncertain model prediction
        eps = 1e-12
        entropy = -float(np.sum(probabilities * np.log2(probabilities + eps)))

        # Get top-k predictions
        top_indices = np.argsort(probabilities)[::-1][:top_k]
        top_preds: List[PredictionItem] = []

        for idx in top_indices:
            idx_int = int(idx)
            prob = float(probabilities[idx_int])
            if self.is_custom and self.custom_classes:
                class_name = self.custom_classes[idx_int] if idx_int < len(self.custom_classes) else f"Class_{idx_int}"
            else:
                class_name = IMAGENET_CATEGORIES[idx_int] if idx_int < len(IMAGENET_CATEGORIES) else f"Class_{idx_int}"

            top_preds.append(PredictionItem(
                class_id=idx_int,
                class_name=class_name,
                probability=round(prob, 4)
            ))

        top_prob = top_preds[0].probability
        second_prob = top_preds[1].probability if len(top_preds) > 1 else 0.0
        margin = round(top_prob - second_prob, 4)

        return ModelPredictions(
            model_name=self.active_version,
            top_predictions=top_preds,
            top_class=top_preds[0].class_name,
            top_probability=top_prob,
            entropy=round(entropy, 3),
            margin=margin,
            embedding_dimension=int(embedding_vector.shape[0]),
            embedding_sample=embedding_sample
        )

# Global singleton classifier instance
_classifier_instance = None

def get_classifier() -> MobileNetV3Classifier:
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = MobileNetV3Classifier()
    return _classifier_instance
