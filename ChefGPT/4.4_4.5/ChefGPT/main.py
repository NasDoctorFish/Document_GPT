from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Nicolacus Maximus Quote Giver",
    description="Get a real quote said by Nicolacus Maximus himself."
)

# Response model for the quote
class Quote(BaseModel):
    quote: str = Field(
        description="The quote that Nicolacus Maximus said."
    )
    year: str = Field(
        description="The year when Nicolacus Maximus said the quote."
    )

# Endpoint to get a random quote
@app.get(
    "/quote",
    summary="Returns a random quote by Nicolacus Maximus",
    description="Upon receiving a GET request, this endpoint will return a real quote said by Nicolacus Maximus himself.",
    response_description="A Quote object that contains the quote said by Nicolacus Maximus and the date when the quote was said.",
    response_model=Quote,
)
def get_quote():
    return Quote(
        quote="Life is short, so eat it all.",
        year="1950"  # Assuming year as a string
    )