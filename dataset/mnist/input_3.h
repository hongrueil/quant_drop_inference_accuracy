#include "../../inference/nn.h"

const fixed data_input_raw[28][28] = {
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.435294151306),
    _Q15(0.945098102093),
    _Q15(1.0),
    _Q15(0.992156922817),
    _Q15(0.647058844566),
    _Q15(0.552941203117),
    _Q15(0.552941203117),
    _Q15(0.235294133425),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.77254909277),
    _Q15(0.988235354424),
    _Q15(0.992156922817),
    _Q15(0.988235354424),
    _Q15(0.988235354424),
    _Q15(0.988235354424),
    _Q15(0.992156922817),
    _Q15(0.917647123337),
    _Q15(0.51372551918),
    _Q15(0.1254902035),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.172549024224),
    _Q15(0.415686309338),
    _Q15(0.611764729023),
    _Q15(0.317647069693),
    _Q15(0.313725501299),
    _Q15(0.415686309338),
    _Q15(0.658823549747),
    _Q15(0.658823549747),
    _Q15(0.917647123337),
    _Q15(0.894117712975),
    _Q15(0.298039227724),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.231372565031),
    _Q15(0.941176533699),
    _Q15(0.933333396912),
    _Q15(0.149019613862),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.494117677212),
    _Q15(0.996078491211),
    _Q15(0.77254909277),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0509803965688),
    _Q15(0.564705908298),
    _Q15(0.95294123888),
    _Q15(0.992156922817),
    _Q15(0.768627524376),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.149019613862),
    _Q15(0.235294133425),
    _Q15(0.819607913494),
    _Q15(0.988235354424),
    _Q15(0.988235354424),
    _Q15(0.905882418156),
    _Q15(0.168627455831),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0509803965688),
    _Q15(0.443137288094),
    _Q15(0.784313797951),
    _Q15(0.929411828518),
    _Q15(0.945098102093),
    _Q15(0.988235354424),
    _Q15(0.988235354424),
    _Q15(0.988235354424),
    _Q15(0.490196108818),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.435294151306),
    _Q15(0.749019622803),
    _Q15(0.996078491211),
    _Q15(0.992156922817),
    _Q15(0.992156922817),
    _Q15(0.992156922817),
    _Q15(0.996078491211),
    _Q15(0.968627512455),
    _Q15(0.968627512455),
    _Q15(0.992156922817),
    _Q15(0.701960802078),
    _Q15(0.0862745121121),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.77254909277),
    _Q15(0.988235354424),
    _Q15(0.992156922817),
    _Q15(0.988235354424),
    _Q15(0.988235354424),
    _Q15(0.792156934738),
    _Q15(0.329411774874),
    _Q15(0.254901975393),
    _Q15(0.258823543787),
    _Q15(0.521568655968),
    _Q15(0.894117712975),
    _Q15(0.917647123337),
    _Q15(0.321568638086),
    _Q15(0.0274509824812),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.623529434204),
    _Q15(0.988235354424),
    _Q15(0.658823549747),
    _Q15(0.658823549747),
    _Q15(0.313725501299),
    _Q15(0.0235294140875),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.172549024224),
    _Q15(0.658823549747),
    _Q15(0.917647123337),
    _Q15(0.454901993275),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0392156876624),
    _Q15(0.109803929925),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.431372582912),
    _Q15(0.894117712975),
    _Q15(0.592156887054),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.447058856487),
    _Q15(0.949019670486),
    _Q15(0.235294133425),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.541176497936),
    _Q15(0.992156922817),
    _Q15(0.768627524376),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.40784317255),
    _Q15(0.964705944061),
    _Q15(0.992156922817),
    _Q15(0.768627524376),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0509803965688),
    _Q15(0.541176497936),
    _Q15(0.964705944061),
    _Q15(0.988235354424),
    _Q15(0.945098102093),
    _Q15(0.231372565031),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0901960805058),
    _Q15(0.0627451017499),
    _Q15(0.113725498319),
    _Q15(0.113725498319),
    _Q15(0.458823561668),
    _Q15(0.552941203117),
    _Q15(0.800000071526),
    _Q15(0.992156922817),
    _Q15(0.992156922817),
    _Q15(0.945098102093),
    _Q15(0.592156887054),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.247058838606),
    _Q15(0.917647123337),
    _Q15(0.843137323856),
    _Q15(0.992156922817),
    _Q15(0.988235354424),
    _Q15(0.988235354424),
    _Q15(0.988235354424),
    _Q15(0.992156922817),
    _Q15(0.988235354424),
    _Q15(0.8156863451),
    _Q15(0.427451014519),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.321568638086),
    _Q15(0.988235354424),
    _Q15(0.988235354424),
    _Q15(0.992156922817),
    _Q15(0.988235354424),
    _Q15(0.988235354424),
    _Q15(0.988235354424),
    _Q15(0.956862807274),
    _Q15(0.364705890417),
    _Q15(0.0509803965688),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.1254902035),
    _Q15(0.894117712975),
    _Q15(0.988235354424),
    _Q15(0.945098102093),
    _Q15(0.54509806633),
    _Q15(0.54509806633),
    _Q15(0.54509806633),
    _Q15(0.0980392247438),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0),
    _Q15(0.0)
};
