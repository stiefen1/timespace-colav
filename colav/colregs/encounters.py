"""
Maritime encounter classification and COLREGS compliance.

Implements the International Regulations for Preventing Collisions at Sea
for encounter classification and collision avoidance recommendations
between vessels in maritime navigation scenarios.

Key Functions
-------------
get_encounter : Classify encounter type between two vessels
get_recommendation_for_os : Get collision avoidance recommendation
get_recommendation_from_encounters : Derive recommendation from encounter types

Key Enums
---------
Encounter : HEAD_ON, STARBOARD, PORT, OVERTAKING encounter types
Recommendation : TURN_RIGHT, TURN_LEFT, DO_NOTHING maneuver recommendations

Notes
-----
Implements COLREGS Rules 13-17 for collision avoidance including:
- Head-on situations (Rule 14)
- Crossing situations (Rule 15) 
- Overtaking situations (Rule 13)
- Good seamanship practices in Traffic Separation Schemes
"""

from enum import Enum
from typing import Tuple, Dict
import numpy as np, logging
from colav.utils.math import wrap_angle_to_pmpi, DEG2RAD, RAD2DEG
from colav.obstacles.moving import MovingShip
logger = logging.getLogger(__name__)

class Encounter(Enum):
    """
    Maritime encounter classification types.
    
    Defines standard encounter situations between vessels
    according to COLREGS regulations.
    
    Values
    ------
    INVALID : Invalid or unclassified encounter
    HEAD_ON : Head-on encounter (Rule 14)
    STARBOARD : Target ship on starboard side (Rule 15)
    PORT : Target ship on port side (Rule 15)
    OVERTAKING : Overtaking situation (Rule 13)
    """
    INVALID = 0
    HEAD_ON = 1
    STARBOARD = 2
    PORT = 3
    OVERTAKING = 4

class Recommendation(Enum):
    """
    Collision avoidance maneuver recommendations.
    
    Defines standard COLREGS-compliant maneuvers for
    collision avoidance between vessels.
    
    Values
    ------
    INVALID : Invalid or no recommendation
    TURN_RIGHT : Turn to starboard (most common)
    DO_NOTHING : Maintain course and speed
    TURN_LEFT : Turn to port (special circumstances)
    """
    INVALID = 0
    TURN_RIGHT = 1
    DO_NOTHING = 2
    TURN_LEFT = 3 # in case of good seamanship when crossing a TSS for instance

CROSSING_ANGLE_DEG = 22.5 # Angle w.r.t horizontal that separate starboard, port from overtaking (degrees)

def get_encounter(
        os: MovingShip, 
        ts: MovingShip,
        head_on_lim_deg:float=10,
        ) -> Encounter:
    """
    Classify encounter type between own ship and target ship.
    
    Determines the type of maritime encounter based on relative
    bearing and COLREGS definitions for collision regulations.
    
    Parameters
    ----------
    os : MovingShip
        Own ship (vessel performing the classification).
    ts : MovingShip
        Target ship (other vessel in the encounter).
    head_on_lim_deg : float, default 10
        Angular limit in degrees for head-on encounter classification.
        
    Returns
    -------
    Encounter
        Classified encounter type between the vessels.
        
    Notes
    -----
    Encounter classification based on relative bearing:
    - HEAD_ON: Target within ±head_on_lim_deg of bow
    - STARBOARD: Target on starboard side (crossing)
    - PORT: Target on port side (crossing)
    - OVERTAKING: Target astern (±22.5° from stern)
    
    Uses COLREGS angular definitions for consistent classification
    across different navigation scenarios.
    
    Examples
    --------
    >>> encounter = get_encounter(own_ship, target_ship)
    >>> if encounter == Encounter.HEAD_ON:
    ...     print("Head-on situation detected")
    """
    heading_os_rad = wrap_angle_to_pmpi(DEG2RAD(os.psi) if os.degrees else os.psi, degrees=False)
    rel_xy = np.array([ts.position[0] - os.position[0], ts.position[1] - os.position[1]])
    
    encounter_angle_raw_rad = np.sign(rel_xy[0]) * np.arccos(rel_xy[1] / np.linalg.norm(rel_xy)) - heading_os_rad
    encounter_angle_raw_deg = RAD2DEG(encounter_angle_raw_rad)
    encounter_angle_deg = wrap_angle_to_pmpi(encounter_angle_raw_deg, degrees=True) # [0, 1] @ pose_rel / (norm([0, 1]) * norm(pose_rel)) = cos(a)

    encounter = Encounter.INVALID
    if -head_on_lim_deg <= encounter_angle_deg <= head_on_lim_deg:
        encounter = Encounter.HEAD_ON
    elif -90-CROSSING_ANGLE_DEG <= encounter_angle_deg <= -head_on_lim_deg:
        encounter = Encounter.PORT
    elif head_on_lim_deg <= encounter_angle_deg <= 90+CROSSING_ANGLE_DEG:
        encounter = Encounter.STARBOARD
    elif (-180 <= encounter_angle_deg <= -90-CROSSING_ANGLE_DEG) or (90+CROSSING_ANGLE_DEG <= encounter_angle_deg <= 180):
        encounter = Encounter.OVERTAKING
    else:
        logger.warning(f"Invalid value for encounter with relative angle {encounter_angle_deg:.2f} deg")

    logger.debug(f"{encounter.name} encounter detected")
    return encounter

def get_recommendation_from_encounters(ts_wrt_os:Encounter, os_wrt_ts:Encounter, good_seamanship:bool=False, ts_in_TSS:bool=False, os_in_TSS:bool=False) -> Recommendation:
    """
    Derive collision avoidance recommendation from encounter classifications.
    
    Implements COLREGS Rules 13-17 to determine appropriate maneuver
    based on mutual encounter classifications and operational context.
    
    Parameters
    ----------
    ts_wrt_os : Encounter
        Target ship encounter type relative to own ship.
    os_wrt_ts : Encounter  
        Own ship encounter type relative to target ship.
    good_seamanship : bool, default False
        Enable good seamanship practices beyond basic COLREGS.
    ts_in_TSS : bool, default False
        Whether target ship is in Traffic Separation Scheme.
    os_in_TSS : bool, default False
        Whether own ship is in Traffic Separation Scheme.
        
    Returns
    -------
    Recommendation
        COLREGS-compliant collision avoidance recommendation.
        
    Notes
    -----
    COLREGS implementation:
    - Rule 13: Overtaking - stand-on vessel maintains course
    - Rule 14: Head-on - both vessels turn to starboard
    - Rule 15: Crossing - give-way vessel turns to starboard
    - Good seamanship: Special considerations for TSS
    
    Priority order:
    1. Overtaking situations (DO_NOTHING)
    2. Good seamanship in TSS (special rules)
    3. Standard COLREGS (typically TURN_RIGHT)
    
    Examples
    --------
    >>> rec = get_recommendation_from_encounters(
    ...     Encounter.STARBOARD, Encounter.PORT
    ... )
    >>> print(f"Recommendation: {rec.name}")
    """
    assert ts_wrt_os != Encounter.INVALID, f"Encounter ts_wrt_os is invalid"
    assert os_wrt_ts != Encounter.INVALID, f"Encounter os_wrt_ts is invalid"
    
    if ts_wrt_os == Encounter.OVERTAKING or os_wrt_ts == Encounter.OVERTAKING:
        recommendation = Recommendation.DO_NOTHING
    
    elif good_seamanship and ts_in_TSS and (not os_in_TSS):
        match ts_wrt_os:
            case Encounter.HEAD_ON:
                    if os_wrt_ts == Encounter.STARBOARD:
                        recommendation = Recommendation.TURN_LEFT
                    else:
                        recommendation = Recommendation.TURN_RIGHT
            case Encounter.STARBOARD:
                if os_wrt_ts == Encounter.STARBOARD:
                    recommendation = Recommendation.TURN_LEFT
                else:
                    recommendation = Recommendation.TURN_RIGHT
            case Encounter.PORT:
                if os_wrt_ts == Encounter.PORT:
                    recommendation = Recommendation.TURN_RIGHT
                else:
                    recommendation = Recommendation.TURN_LEFT
            case _:
                logger.warning(f"Invalid Encounter value {ts_wrt_os}")
                recommendation = Recommendation.INVALID
    else:
        match ts_wrt_os:
            case Encounter.HEAD_ON:
                    recommendation = Recommendation.TURN_RIGHT
            case Encounter.STARBOARD:
                if os_wrt_ts == Encounter.STARBOARD:
                    recommendation = Recommendation.DO_NOTHING
                else:
                    recommendation = Recommendation.TURN_RIGHT
            case Encounter.PORT:
                if os_wrt_ts == Encounter.PORT:
                    recommendation = Recommendation.DO_NOTHING
                else:
                    recommendation = Recommendation.TURN_RIGHT
            case _:
                logger.warning(f"Invalid Encounter value {ts_wrt_os}")
                recommendation = Recommendation.INVALID

    logger.debug(f"Recommended {recommendation.name}")
    return recommendation

def get_recommendation_for_os(
        os: MovingShip, 
        ts: MovingShip,
        head_on_lim_deg: float = 10,
        good_seamanship: bool = False,
        ts_in_TSS: bool = False,
        os_in_TSS: bool = False
    ) -> Tuple[Recommendation, Dict]:
    """
    Get COLREGS-compliant collision avoidance recommendation for own ship.
    
    Complete COLREGS analysis combining encounter classification with
    regulatory compliance to provide collision avoidance recommendation
    for the own ship in a two-vessel encounter.
    
    Parameters
    ----------
    os : MovingShip
        Own ship requiring collision avoidance recommendation.
    ts : MovingShip
        Target ship in the encounter.
    head_on_lim_deg : float, default 10
        Angular limit for head-on encounter classification.
    good_seamanship : bool, default False
        Enable good seamanship practices beyond basic COLREGS.
    ts_in_TSS : bool, default False
        Whether target ship operates in Traffic Separation Scheme.
    os_in_TSS : bool, default False
        Whether own ship operates in Traffic Separation Scheme.
        
    Returns
    -------
    recommendation : Recommendation
        COLREGS-compliant collision avoidance maneuver.
    info : dict
        Analysis information containing:
        - 'encounter_ts_wrt_os': Target ship encounter classification
        - 'encounter_os_wrt_ts': Own ship encounter classification
        
    Notes
    -----
    Complete COLREGS implementation workflow:
    1. Classify encounter from both vessel perspectives
    2. Apply COLREGS Rules 13-17 for collision avoidance
    3. Consider good seamanship and TSS operations
    4. Return recommendation with supporting analysis
    
    Typical recommendations:
    - TURN_RIGHT: Most common COLREGS maneuver
    - DO_NOTHING: Stand-on vessel or overtaking situation
    - TURN_LEFT: Special good seamanship circumstances
    
    Examples
    --------
    Basic COLREGS analysis:
    
    >>> recommendation, info = get_recommendation_for_os(
    ...     own_ship, target_ship
    ... )
    >>> print(f"Maneuver: {recommendation.name}")
    >>> print(f"Encounter: {info['encounter_ts_wrt_os'].name}")
    
    With Traffic Separation Scheme:
    
    >>> rec, info = get_recommendation_for_os(
    ...     own_ship, target_ship,
    ...     good_seamanship=True,
    ...     ts_in_TSS=True,
    ...     os_in_TSS=False
    ... )
    
    Head-on with custom angle:
    
    >>> rec, info = get_recommendation_for_os(
    ...     own_ship, target_ship,
    ...     head_on_lim_deg=15  # Wider head-on definition
    ... )
    
    See Also
    --------
    get_encounter : Individual encounter classification
    get_recommendation_from_encounters : Core recommendation logic
    Encounter : Encounter type definitions
    Recommendation : Maneuver recommendation types
    """
    encounter_ts_wrt_os = get_encounter(os, ts, head_on_lim_deg=head_on_lim_deg)
    encounter_os_wrt_ts = get_encounter(ts, os, head_on_lim_deg=head_on_lim_deg)
    info = {
        'encounter_ts_wrt_os': encounter_ts_wrt_os,
        'encounter_os_wrt_ts': encounter_os_wrt_ts
    }
    return get_recommendation_from_encounters(encounter_ts_wrt_os, encounter_os_wrt_ts, good_seamanship=good_seamanship, ts_in_TSS=ts_in_TSS, os_in_TSS=os_in_TSS), info