"""Machine learning functions."""

"""Adding another line with Azamt"""

import logging

from joblib import load
from fastapi import APIRouter
import pandas as pd
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)
router = APIRouter()

# uvicorn app.main:app --reload

classifier = load("app/classifier.joblib")


class Item(BaseModel):
    """Use this data model to parse the request body JSON."""

    category: str = Field(..., example="Food")
    main_category: str = Field(..., example="Drink")
    backers: int = Field(..., example=25)
    usd_goal_real: float = Field(..., example=5000.00)
    ks_length: int = Field(..., example=60)

    def to_df(self):
        """Convert pydantic object to pandas dataframe with 1 row."""
        return pd.DataFrame([dict(self)])

    # @validator('x3')
    # def x3_must_be_positive(cls, value):
    #     """Validate that x3 is a positive number."""
    #     assert value > 0, f'x3 == {value}, must be > 0'
    #     return value


@router.post("/predict")
async def predict(item: Item):
    """
    Make random baseline predictions for classification problem 🔮.

    ### Request Body
    - `category`: Select a category ['Poetry', 'Narrative Film', 'Music',
       'Restaurants', 'Food',
       'Drinks', 'Nonfiction', 'Indie Rock', 'Crafts', 'Games',
       'Tabletop Games', 'Design', 'Comic Books', 'Art Books', 'Fashion',
       'Childrenswear', 'Theater', 'Comics', 'DIY', 'Webseries',
       'Animation', 'Food Trucks', 'Product Design', 'Public Art',
       'Documentary', 'Illustration', 'Photography', 'Pop', 'People',
       'Art', 'Family', 'Fiction', 'Film & Video', 'Accessories', 'Rock',
       'Hardware', 'Software', 'Weaving', 'Web', 'Jazz', 'Ready-to-wear',
       'Festivals', 'Video Games', 'Anthologies', 'Publishing', 'Shorts',
       'Gadgets', 'Electronic Music', 'Radio & Podcasts', 'Cookbooks',
       'Apparel', 'Metal', 'Comedy', 'Hip-Hop', 'Periodicals', 'Dance',
       'Technology', 'Painting', 'World Music', 'Photobooks', 'Drama',
       'Architecture', 'Young Adult', 'Latin', 'Mobile Games', 'Flight',
       'Fine Art', 'Action', 'Playing Cards', 'Makerspaces', 'Punk',
       "Children's Books", 'Apps', 'Audio', 'Performance Art', 'Ceramics',
       'Vegan', 'Graphic Novels', 'Fabrication Tools', 'Performances',
       'Sculpture', 'Sound', 'Stationery', 'Print', "Farmer's Markets",
       'Thrillers', 'Events', 'Classical Music', 'Graphic Design',
       'Spaces', 'Country & Folk', 'Wearables', 'Journalism',
       'Mixed Media', 'Movie Theaters', 'Animals', 'Digital Art',
       'Knitting', 'Installations', 'Community Gardens',
       'DIY Electronics', 'Embroidery', 'Camera Equipment', 'Jewelry',
       'Farms', 'Conceptual Art', 'Fantasy', 'Webcomics', 'Horror',
       'Experimental', 'Science Fiction', 'Puzzles', 'R&B',
       'Music Videos', 'Video', 'Plays', 'Blues', 'Bacon', 'Faith',
       'Live Games', 'Small Batch', 'Woodworking', 'Places', 'Footwear',
       '3D Printing', 'Zines', 'Musical', 'Workshops', 'Photo',
       'Immersive', 'Letterpress', 'Academic', 'Candles', 'Television',
       'Space Exploration', 'Gaming Hardware', 'Nature', 'Robots',
       'Typography', 'Translations', 'Calendars', 'Textiles', 'Pottery',
       'Interactive Design', 'Video Art', 'Glass', 'Pet Fashion',
       'Crochet', 'Printing', 'Romance', 'Civic Design', 'Kids',
       'Literary Journals', 'Couture', 'Taxidermy', 'Quilts', 'Chiptune',
       'Residencies', 'Literary Spaces']

    - `main_category`: Select a main category ['Publishing', 'Film & Video',
       'Music', 'Food', 'Design', 'Crafts',
       'Games', 'Comics', 'Fashion', 'Theater', 'Art', 'Photography',
       'Technology', 'Dance', 'Journalism']

    - `backers`: Estimate the number of expected backers for your Kickstarter
        project

    - `usd_goal_real`: Enter prospective Kickstarter campaign's fundraising
        goal in U.S. dollars

    - `ks_length`: Enter the duration of proposed Kickstarter fundraising
        campaign in days

    ### Response
    - `prediction`: success, failed
    - `predict_proba`: float between 0.0 and 1.0,
    representing the predicted class's probability

    """

    X_new = item.to_df()
    choice = classifier.predict(X_new)
    probability = classifier.predict_proba(X_new)
    return choice[0], f"{probability[0][1]*100:.2f}% probability"
