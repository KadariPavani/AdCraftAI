# Ads from the web 
Our `OnlineAds.zip` comprise images scraped from platforms such as Google Images, Instagram, and Facebook. These ad images are arranged hierarchically within nested directories for efficient organization and accessibility. Unzip the downloaded file to access the online ads folder, which can be used for training your models and other purposes.

The following is the arrangement ad images in OnlineAds.

    .
    └── OnlineAds
        ├── baby_products
        ├── body_wear
        │   ├── clothing_brands
        │   ├── footwear
        │   ├── jewellery
        │   └── watch
        │       └── Casio_watches
        │           ├── Casio_watches_1.jpg
        │           ├── Casio_watches_2.jpg
        │           ├── ...
        │           └── Casio_watches_62.jpg
        ├── cosmetics
        ├── ... 
        ├── travel
        └── vehicles

The ads within OnlineAds follow a specific naming convention: they begin with the brand name followed by a numerical identifier. 

The annotations for these ads are present in `web_annot_j.json`, in the following format

```
{
    "img_path": "..\\OnlineAds\\body_wear\\watch\\Casio_watches\\Casio_watches_1.jpg",
    "hier_annot": [
        "body_wear",
        "watch",
        "Casio_watches"
    ],
    "language": "english",
    "source": "internet",
    "ad_type": "product"
},
```

# Newspaper Advertisements

## Advert_Gallery.zip
This contains ads scraped from the Advert Gallery. Unlike online ads, they aren't arranged hierarchically, but instead have folders of various brands, which are named exaclty as those found in online ads. Additionally, the script `merge_adgal_and_ads.py` is provided to merge both OnlineAds and Advert_Gallery. 

The following example demostrates the structure of Advert_Gallery

    .
    └── Advert_Gallery
        └── NewsPaperAds
            └── Advert_Gallery
                ├── Aashirvad_Atta
                ├── Adidas
                │   ├── Adidas_400.jpg
                │   ├── Adidas_401.jpg
                │   └── ...
                ├── Air_India
                ├── Airtel
                ├── ...
                ├── Yes_Bank
                └── Zomato

Similar to ads from OnlineAds, those in Advert_Gallery adhere to a naming convention: they begin with the brand name followed by a numerical identifier, which starts from 400. There's no particular significance to starting from 400, except for maintaining distinct image names.

The annotations for these are present in `adgal_annot_j.json`, in the following format

```
{
    "img_path": "../NewsPaperAds/Advert_Gallery\\Adidas\\Adidas_401.jpg",
    "hier_annot": [
        "sports",
        "sports_apparel",
        "Adidas"
    ],
    "language": "english",
    "source": "newspaper",
    "ad_type": "product"
},
```

## Epaper1

In the folder Epaper1, there are subfolders corresponding to each language. Within each language folder, the naming convention is as follows: <Newspaper_A>_1.jpg, <Newspaper_A>_2.jpg, and so on, for ads of Newspaper_A. 

### Languages:
- Assamese
- Urdu
- Odia
- Gujarati
- Punjabi
- Hindi
- Marathi
- Bengali

### Dataset Structure

The Epaper1 dataset follows the following folder structure:


    Epaper1/  
    │  
    ├── Assamese/  
    │   ├── Newspaper_1.jpg  
    │   ├── Newspaper_2.jpg  
    │   └── ...  
    │  
    ├── Urdu/  
    │   ├── Newspaper_1.jpg  
    │   ├── Newspaper_2.jpg  
    │   └── ...  
    │  
    ├── Odia/   
    │   ├── Newspaper_1.jpg  
    │   ├── Newspaper_2.jpg  
    │   └── ...  
    │  
    ├── Gujarati/  
    │   ├── Newspaper_1.jpg  
    │   ├── Newspaper_2.jpg  
    │   └── ...  
    │  
    ├── Punjabi/  
    │   ├── Newspaper_1.jpg  
    │   ├── Newspaper_2.jpg  
    │   └── ...  
    │  
    ├── Hindi/  
    │   ├── Newspaper_1.jpg  
    │   ├── Newspaper_2.jpg  
    │   └── ...  
    │  
    ├── Marathi/  
    │   ├── Newspaper_1.jpg  
    │   ├── Newspaper_2.jpg  
    │   └── ...  
    │  
    └── Bengali/  
    ├── Newspaper_1.jpg  
    ├── Newspaper_2.jpg  
    └── ...  

Structure of annotations present in `epaper1_annotation.json`

```
    {
        "img_path": "../Epaper1/Assamese/The Sangai Express_1366.jpg",
        "source": "newpaper",
        "language": "Assamese"
    },
```



## Epaper2 
Similar to Epaper1, Epaper2 has subfolders corresponding to each language. Within each language folder, the naming convention is as follows: <Newspaper_A>_1.jpg, <Newspaper_A>_2.jpg, and so on, for ads of Newspaper_A. 

### Languages:
- Urdu
- Odia
- Telugu
- Gujarati
- Hindi
- Tamil
- Kannada
- Malayalam
- Marathi
- Bengali

### Dataset Structure

The Epaper2 dataset follows the same folder structure as Epaper1:

    Epaper2/  
    │  
    ├── Urdu/  
    │   ├── Newspaper_1.jpg  
    │   ├── Newspaper_2.jpg  
    │   └── ...  
    │  
    ├── Odia/  
    │   ├── Newspaper_1.jpg  
    │   ├── Newspaper_2.jpg  
    │   └── ...  
    │  
    ├── Telugu/  
    │   ├── Newspaper_1.jpg  
    │   ├── Newspaper_2.jpg  
    │   └── ...  
    │  
    ├── Gujarati/  
    │   ├── Newspaper_1.jpg  
    │   ├── Newspaper_2.jpg  
    │   └── ...  
    │  
    ├── Hindi/  
    │   ├── Newspaper_1.jpg  
    │   ├── Newspaper_2.jpg  
    │   └── ...  
    │  
    ├── Tamil/  
    │   ├── Newspaper_1.jpg  
    │   ├── Newspaper_2.jpg  
    │   └── ...  
    │  
    ├── Kannada/  
    │   ├── Newspaper_1.jpg  
    │   ├── Newspaper_2.jpg  
    │   └── ...  
    │  
    ├── Malayalam/  
    │   ├── Newspaper_1.jpg  
    │   ├── Newspaper_2.jpg  
    │   └── ...  
    │  
    ├── Marathi/  
    │   ├── Newspaper_1.jpg  
    │   ├── Newspaper_2.jpg  
    │   └── ...  
    │  
    └── Bengali/  
        ├── Newspaper_1.jpg  
        ├── Newspaper_2.jpg  
        └── ...  


Structure of annotations present in `epaper2_annotation.json`
```
{
    "img_path": "../Epaper2/Urdu/Inquilab_85.jpg",
    "source": "newpaper",
    "language": "Urdu"
},
{
    "img_path": "../Epaper2/Urdu/Inquilab_206.jpg",
    "source": "newpaper",
    "language": "Urdu"
},

```