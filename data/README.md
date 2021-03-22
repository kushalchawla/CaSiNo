
This directory contains the CaSiNo dataset along with the associated strategy annotations.

# File Descriptions

**casino_complete.json**: The complete set of 1030 dialogues in the CaSiNo dataset.\
**casino_annotated.json**: The CaSiNo-Ann dataset containing the strategy annotations for a subset of 396 dialogues. The dialogues in this file can be mapped to their complete information in casino_complete.json using the dialogue_id key.

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

If you want to work with the participant demographics and personality attributes, please email us at kchawla@usc.edu
