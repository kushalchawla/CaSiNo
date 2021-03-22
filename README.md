# CaSiNo

This repository contains the dataset and the PyTorch code for **'CaSiNo: A Corpus of Campsite Negotiation Dialogues for Automatic Negotiation Systems'**.

We provide a novel dataset (referred to as CaSiNo) of 1030 negotiation dialogues. Two participates take the role of campsite neighbors and negotiate for *Food*, *Water*, and *Firewood* packages, based on their individual preferences and requirements. This design keeps the task tractable, while still facilitating linguistically rich and personal conversations.

# Repository Structure

**data**: The complete CaSiNo dataset and the strategy annotations.\
**strategy_prediction**: Code for strategy prediction in a multi-task learning setup.

# Each Dialogue in the Dataset

**Participant Info**
* Demographics (Age, Gender, Ethnicity, Education)
* Personality attributes (SVO and Big-5)
* Preference order
* Arguments for needing or not needing a specific item

**Negotiation Dialogue**
* Alternating conversation between two participants
* 11.6 utterances on average
* Includes the use of four emoticons: Joy, Sadness, Anger, Surprise

**Negotiation Outcomes**
* Points scored
* Satisfaction (How satisfied are you with the negotiation outcome?)
* Opponent Likeness (How much do you like your opponent?)

# References

To Be Added

# LICENSE

Please refer to the LICENSE file in the root repo folder for more details.
