# ----------------------------------
# Browser
# ----------------------------------

HEADLESS = True

# ----------------------------------
# Scraper
# ----------------------------------

MAX_PAGES = 6

RETRY_COUNT = 3

CHECKPOINT_EVERY = 50

DELAY_MIN = 800
DELAY_MAX = 1800

# ----------------------------------
# Output
# ----------------------------------

OUTPUT_DIR = "../data/raw"

# ----------------------------------
# Cars to scrape
# ----------------------------------

CARS = [

    # Suzuki
    
    
  
    

    # Toyota
    
    ("Toyota","Yaris"),
    ("Toyota","Hilux"),
    ("Toyota","Fortuner"),
    ("Toyota","Corolla Cross"),
    ("Toyota","Aqua"),

    # Honda
    ("Honda","City"),
    ("Honda","Civic"),
    ("Honda","BR-V"),
    ("Honda","HR-V"),

    # Kia
    ("Kia","Picanto"),
    ("Kia","Sportage"),
    ("Kia","Stonic"),
    ("Kia","Sorento"),

    # Hyundai
    ("Hyundai","Elantra"),
    ("Hyundai","Sonata"),
    ("Hyundai","Tucson"),
    ("Hyundai","Santa Fe"),

    # Changan
    ("Changan","Alsvin"),
    ("Changan","Oshan X7"),
    ("Changan","Karvaan"),

    # MG
    ("MG","HS"),
    ("MG","ZS EV"),

    # Proton
    ("Proton","Saga"),
    ("Proton","X70"),

]
SAVE_CHECKPOINT = True

SAVE_EXCEL = True

SAVE_CSV = True

LOG_FILE = "../logs/scraper.log"