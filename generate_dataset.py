"""
generate_dataset.py
-------------------
Generates synthetic true.csv and fake.csv datasets for training.
Creates ~600 samples each with realistic news patterns.

Run with:
    python3.13 generate_dataset.py
"""

import csv
import random
import os

random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ #
#  REAL NEWS TEMPLATES
#  Realistic factual-style sentences covering politics, science,
#  economics, health, world affairs, technology.
# ------------------------------------------------------------------ #

REAL_TITLES = [
    "Senate passes bipartisan infrastructure bill with broad support",
    "WHO releases updated guidelines on COVID-19 booster vaccines",
    "Federal Reserve raises interest rates by 25 basis points",
    "NASA's James Webb Telescope captures earliest galaxies ever recorded",
    "Climate summit reaches new agreement on carbon emission targets",
    "Supreme Court rules on landmark voting rights case",
    "Global food prices rise for third consecutive quarter, UN report says",
    "Tech companies agree to new data privacy standards in EU",
    "Scientists discover potential new treatment for Alzheimer's disease",
    "Stock markets recover after weeks of volatility amid rate hike fears",
    "United Nations calls for immediate ceasefire in ongoing conflict",
    "President signs executive order on renewable energy subsidies",
    "New study links air pollution to increased risk of heart disease",
    "G7 nations pledge additional aid to developing countries",
    "Major oil producers agree to cut output amid falling prices",
    "Health officials confirm outbreak of rare respiratory illness contained",
    "Federal court upholds net neutrality regulations",
    "International space station crew completes emergency repair walk",
    "Economy adds 250,000 jobs in latest monthly report",
    "Researchers develop breakthrough battery technology for electric vehicles",
    "Congress approves funding for new national broadband initiative",
    "Central bank holds rates steady amid inflation concerns",
    "Tropical storm strengthens to category three hurricane near coast",
    "Secretary of state meets with foreign counterparts on trade deal",
    "New data shows child poverty rate fell to historic low last year",
    "Scientists confirm record-breaking Antarctic ice melt season",
    "Medical journal publishes findings on mRNA vaccine long-term safety",
    "Tech giant faces antitrust scrutiny from multiple regulatory bodies",
    "Economists warn of slowing growth in emerging market economies",
    "Defense department announces new cybersecurity strategy",
    "Census bureau releases updated population growth figures",
    "World bank approves loan package for infrastructure development",
    "Local elections show shift in voter preferences toward independents",
    "Public health agency reports decline in smoking rates nationwide",
    "Satellite images confirm deforestation in protected rainforest areas",
    "Scientists map complete human genome with new sequencing technology",
    "Drug regulator approves first gene therapy for inherited blindness",
    "Consumer confidence index rises to highest level in three years",
    "United states and allies impose new sanctions over human rights concerns",
    "Report shows renewable energy now cheaper than fossil fuels in most regions",
    "Government launches task force to investigate housing affordability crisis",
    "International criminal court issues arrest warrant for war crimes",
    "Unemployment rate falls to lowest level since pandemic",
    "Agricultural research station develops drought-resistant crop strains",
    "Study finds Mediterranean diet reduces risk of cardiovascular disease",
    "State department issues travel advisory for several regions",
    "Audit finds government agency misused funds, reform measures announced",
    "Space agency successfully tests next-generation rocket propulsion system",
    "New international treaty on ocean plastic pollution signed by forty nations",
    "Economic report shows inflation easing for fourth consecutive month",
]

REAL_BODIES = [
    "The legislation was passed after months of negotiations between both parties. Officials confirmed the measure will take effect following presidential signature.",
    "Health authorities emphasized that the updated protocols are based on the latest peer-reviewed clinical data and represent consensus among leading experts.",
    "The decision reflects ongoing concerns about persistent inflation despite recent cooling in consumer price data released by the bureau of labor statistics.",
    "Researchers at the institution said the findings represent a major step forward and published their results in the peer-reviewed journal Nature.",
    "Representatives from over 190 countries attended the conference, with negotiators working through the night to finalize the binding agreement.",
    "The ruling, written by the majority opinion, cited constitutional protections and precedent established over the past several decades.",
    "Official government data released Thursday confirmed the trend, with analysts noting that supply chain improvements had begun to ease pressures.",
    "The agency's spokesperson confirmed the decision in a formal statement, noting that all regulatory requirements had been met following an extensive review.",
    "Independent experts said the study, which followed more than 50,000 participants over five years, provides robust evidence for the proposed link.",
    "Market analysts attributed the rebound to stronger-than-expected earnings reports and optimism about a potential pause in monetary tightening.",
    "The organization's secretary general called on all parties to exercise restraint and return to the negotiating table without preconditions.",
    "Congressional budget office estimates indicate the measure will cost approximately 400 billion dollars over the next decade.",
    "The research, conducted across 12 hospitals and reviewed by independent scientists, represents one of the largest studies of its kind.",
    "Finance ministers from member nations agreed on the package after two days of intensive talks at the annual meeting in Geneva.",
    "The cartel's decision to reduce output by 1.5 million barrels per day came amid concerns that prices had fallen below sustainable levels.",
    "Public health officials said containment measures had proved effective and thanked healthcare workers for their rapid response.",
    "The three-judge panel ruled unanimously, upholding the regulations as consistent with the administrative procedures act.",
    "Mission control confirmed the six-hour spacewalk was completed successfully and all station systems were functioning normally.",
    "The labor department report exceeded expectations, with gains spread across construction, healthcare, and professional services sectors.",
    "The technology, described in a new paper in the journal Science, could more than double the energy density of current lithium-ion batteries.",
    "The bipartisan bill passed with support from senators in both parties, reflecting broad agreement on the need to expand access to high-speed internet.",
    "Committee members voted unanimously to hold rates at the current target range, citing ongoing uncertainty about the economic outlook.",
    "The national hurricane center warned residents in affected areas to follow evacuation orders and prepare for storm surge and heavy rainfall.",
    "The talks focused on reducing tariffs and resolving longstanding disputes over agricultural subsidies that have strained trade relations.",
    "The census bureau credited improvements in the earned income tax credit and expanded child tax credit with reducing family poverty rates.",
    "Satellite data analyzed by researchers shows the ice sheet lost more mass in the past year than in any previously recorded twelve-month period.",
    "The study, which followed participants for more than two years post-vaccination, found no significant increase in adverse events.",
    "Regulators in three jurisdictions have opened parallel investigations into whether the company abused its dominant market position.",
    "The international monetary fund revised its growth forecasts downward for several economies amid rising debt levels and weaker export demand.",
    "The framework outlines priorities including critical infrastructure protection, workforce development, and public-private partnerships.",
]

# ------------------------------------------------------------------ #
#  FAKE NEWS TEMPLATES
#  Sensationalist, misleading, conspiracy-style language patterns.
# ------------------------------------------------------------------ #

FAKE_TITLES = [
    "SHOCKING: Government secretly putting microchips in drinking water revealed",
    "BREAKING: Celebrity admits to being part of global shadow organization",
    "Scientists ADMIT vaccines contain hidden DNA-altering substances",
    "Whistleblower exposes MASSIVE fraud cover-up the media won't report",
    "EXPOSED: Chemtrails proven to be mind control program by leaked documents",
    "Major bank secretly erasing millions of accounts without warning",
    "CONFIRMED: Moon landing was filmed in secret studio, new evidence shows",
    "URGENT: New 5G towers linked to mysterious illness spreading rapidly",
    "Deep state operatives caught rigging election machines across 47 states",
    "Bombshell: Politician caught on tape confessing to secret global deal",
    "They don't want you to know this cancer cure they've been hiding for years",
    "ALERT: Foreign nation has already hacked every US bank account",
    "Actor reveals Hollywood elite sacrifice rituals in explosive new interview",
    "LEAKED DOCUMENT: Government planned pandemic to control world population",
    "Scientists bribed to hide proof that the earth is actually flat",
    "BREAKING: New law will allow government to seize all private firearms this week",
    "Miracle herb cures all disease but big pharma is suppressing the truth",
    "EXPOSED: Famous politician is actually a reptilian shapeshifter with proof",
    "Mass graves found at secret underground facility near major US city",
    "ALERT: Grocery stores will be EMPTY within 72 hours say insiders",
    "They are putting chemicals in food to make people infertile by 2030",
    "Secret memo proves moon is artificial satellite built by ancient civilization",
    "BREAKING: All major news anchors are reading from same CIA-written script",
    "Doctor fired for telling truth about dangerous hidden ingredient in flu shots",
    "URGENT WARNING: New digital currency will delete all cash savings overnight",
    "Insider reveals plan to replace world governments with single global authority",
    "SHOCK REPORT: Billionaires have already moved to secret bunkers for collapse",
    "They are spraying the air to give everyone memory loss, expert confirms",
    "CONFIRMED: Voting machines automatically switch votes in critical districts",
    "Famous scientist admits climate change is hoax to steal taxpayer money",
    "Exclusive: Elite politicians attend secret meetings to plan economic crash",
    "BREAKING: Government to begin mandatory tracking of all citizens within days",
    "Proof emerges that major pharmaceutical company knew drug caused cancer",
    "ALERT: Strange creature spotted near nuclear plant covered up by officials",
    "They are hiding the cure for all viruses to keep pharmaceutical profits high",
    "Secret society controls every major world leader through blackmail operation",
    "EXPOSED: Entire mainstream media is owned by three shadowy families",
    "BREAKING: Alien spacecraft discovered under arctic ice, sources confirm",
    "Government insider leaks plan to eliminate social security completely",
    "URGENT: Hospitals ordered to hide true death toll from secret new disease",
    "New world order document leaked showing plan to reduce world population",
    "SHOCK CLAIM: Famous person faked their own death, spotted in new country",
    "They are poisoning the rain to depopulate rural communities says whistleblower",
    "CONFIRMED: Major tech companies spying on you through turned-off devices",
    "Secret law passed overnight giving government total control of internet",
    "BREAKING: Central bank plans to confiscate gold and silver from citizens",
    "Insider exposes how all elections in western countries are pre-determined",
    "ALERT: Deadly parasite found in common food product linked to thousands of deaths",
    "Government scientist admits weather control technology being used against citizens",
    "EXPOSED: World leaders have secret underground city to survive coming catastrophe",
]

FAKE_BODIES = [
    "Sources who wish to remain anonymous have come forward with explosive information that the mainstream media refuses to cover because of their connections to the elite.",
    "A leaked document that has gone viral on social media appears to confirm what many have long suspected, though officials deny any knowledge of the claims.",
    "The whistleblower, who claims to have worked inside the organization for twenty years, says the truth has been systematically suppressed to protect powerful interests.",
    "Independent researchers who have been silenced by the establishment have finally released their bombshell findings that the government doesn't want you to see.",
    "Despite a complete media blackout, the evidence is overwhelming and thousands of people are waking up to what has really been going on behind closed doors.",
    "An anonymous government insider passed documents to our reporter showing the scale of deception that goes all the way to the highest levels of power.",
    "The real story is being censored across all platforms but brave truth tellers are fighting back against the globalist agenda to control the population.",
    "Eye witnesses report seeing government vehicles at the location while officials claim nothing unusual happened, raising serious questions about the cover-up.",
    "The so-called experts are lying to protect their funding from billionaire donors who are pulling the strings of every major institution in the world.",
    "Share this before they delete it because they are already scrubbing this information from the internet to keep the public in the dark about what is coming.",
    "Those who dare to speak out are immediately discredited, fired from their jobs, and silenced by a coordinated campaign run by shadowy figures in the background.",
    "The truth is hidden in plain sight but the population has been conditioned through years of false information to not question the official narrative.",
    "People are waking up in record numbers and refusing to believe the lies pushed by the globalists who control every aspect of modern life and media.",
    "A secret network of powerful individuals has been running this operation for decades and the noose is finally closing as more brave insiders come forward.",
    "Do your own research because the mainstream media has been completely compromised and will never report on the shocking reality of what is happening.",
    "Doctors and nurses across the country are secretly sharing this information because they are afraid to go public due to threats from hospital management.",
    "The plan has been documented in numerous leaked emails and internal communications that powerful interests are trying to erase from the public record.",
    "This has been going on for over fifty years but technology now makes it impossible for them to keep the lid on their massive global deception operation.",
    "Multiple independent sources from different countries have now confirmed the same disturbing details about this coordinated worldwide cover-up scheme.",
    "The elite are scared because too many people are waking up and asking questions they cannot answer without revealing the true depth of the conspiracy.",
]

def generate_text(titles, bodies, n=600):
    rows = []
    for i in range(n):
        title = random.choice(titles)
        # Combine 2-4 body sentences for variety
        num_sentences = random.randint(2, 4)
        body = " ".join(random.choice(bodies) for _ in range(num_sentences))
        # Add some title variation
        suffix_words = ["report", "sources say", "officials confirm", "insiders reveal", "data shows"]
        if random.random() > 0.5:
            title = title + ", " + random.choice(suffix_words)
        rows.append({"title": title, "text": body})
    return rows


def main():
    true_path = os.path.join(BASE_DIR, "true.csv")
    fake_path = os.path.join(BASE_DIR, "fake.csv")

    print("[INFO] Generating true.csv ...")
    true_rows = generate_text(REAL_TITLES, REAL_BODIES, n=700)
    with open(true_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "text"])
        writer.writeheader()
        writer.writerows(true_rows)
    print(f"[INFO] Wrote {len(true_rows)} rows -> {true_path}")

    print("[INFO] Generating fake.csv ...")
    fake_rows = generate_text(FAKE_TITLES, FAKE_BODIES, n=700)
    with open(fake_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "text"])
        writer.writeheader()
        writer.writerows(fake_rows)
    print(f"[INFO] Wrote {len(fake_rows)} rows -> {fake_path}")

    print("\n[SUCCESS] Datasets ready. Now run: python3.13 train_model.py")


if __name__ == "__main__":
    main()
