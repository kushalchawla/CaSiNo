
This directory contains the CaSiNo dataset along with the associated strategy annotations.

# Descriptions

**casino.json**: The complete set of 1030 dialogues in the CaSiNo dataset, containing the conversations, participant information, and strategy annotations, wherever available.\
**split**: Contains one random split of casino.json into train/valid/test.

# Format for the outcome variables

*points_scored*
 * Points gained for one single item are based on these rules - every high item: 5 , every medium item: 4, every low item: 3
 * Final points can be computed by simply summing up the points for all the items that the participant is able to negotiate for, as per the final agreed deal.
 * If someone walks away, the final points are equal to 5 (equivalent of one high item), for both the participants.

*satisfaction (How satisfied are you with the negotiation outcome?)*
 * 5 possible values (Extremely dissatisfied, Slightly dissatisfied, Undecided, Slightly satisfied, Extremely satisfied)
 * For the analysis in the paper, these were encoded on a scale from 1 to 5 (with increasing satisfaction)

*opponent_likeness (How much do you like your opponent?)*
* 5 possible values (Extremely dislike, Slightly dislike, Undecided, Slightly like, Extremely like)
* For the analysis in the paper, these were encoded on a scale from 1 to 5 (with increasing likeness)

# Notes

For more information, please refer to the original dataset paper published at NAACL 2021 (https://aclanthology.org/2021.naacl-main.254.pdf).

You can also reach out to us at kchawla@usc.edu (Kushal Chawla).
