from PIL import Image

# src:
#   https://gitlab.aicrowd.com/flatland/flatland/blob/master/flatland/utils/graphics_pil.py
#   https://gitlab.aicrowd.com/flatland/flatland/blob/master/flatland/core/grid/grid4.py

image_dir = "static/flatland_png/"
rail_files = {
    "": "Background_white.png",
    "WE": "Gleis_Deadend.png",
    "WW EE NN SS": "Gleis_Diamond_Crossing.png",
    "WW EE": "Gleis_horizontal.png",
    "EN SW": "Gleis_Kurve_oben_links.png",
    "WN SE": "Gleis_Kurve_oben_rechts.png",
    "ES NW": "Gleis_Kurve_unten_links.png",
    "NE WS": "Gleis_Kurve_unten_rechts.png",
    "NN SS": "Gleis_vertikal.png",
    "NN SS EE WW ES NW SE WN": "Weiche_Double_Slip.png",
    "EE WW EN SW": "Weiche_horizontal_oben_links.png",
    "EE WW SE WN": "Weiche_horizontal_oben_rechts.png",
    "EE WW ES NW": "Weiche_horizontal_unten_links.png",
    "EE WW NE WS": "Weiche_horizontal_unten_rechts.png",
    "NN SS EE WW NW ES": "Weiche_Single_Slip.png",
    "NE NW ES WS": "Weiche_Symetrical.png",
    "NN SS EN SW": "Weiche_vertikal_oben_links.png",
    "NN SS SE WN": "Weiche_vertikal_oben_rechts.png",
    "NN SS NW ES": "Weiche_vertikal_unten_links.png",
    "NN SS NE WS": "Weiche_vertikal_unten_rechts.png",
    "NE NW ES WS SS NN": "Weiche_Symetrical_gerade.png",
    "NE EN SW WS": "Gleis_Kurve_oben_links_unten_rechts.png"
}
target_files = {
            "EW": "Bahnhof_#d50000_Deadend_links.png",
            "NS": "Bahnhof_#d50000_Deadend_oben.png",
            "WE": "Bahnhof_#d50000_Deadend_rechts.png",
            "SN": "Bahnhof_#d50000_Deadend_unten.png",
            "EE WW": "Bahnhof_#d50000_Gleis_horizontal.png",
            "NN SS": "Bahnhof_#d50000_Gleis_vertikal.png"}


def get_transitions(cell_transition, orientation):
    """
    Get the 4 possible transitions ((N,E,S,W), 4 elements tuple
    if no diagonal transitions allowed) available for an agent oriented
    in direction `orientation` and inside a cell with
    transitions `cell_transition`.

    Parameters
    ----------
    cell_transition : int
        16 bits used to encode the valid transitions for a cell.
    orientation : int
        Orientation of the agent inside the cell.

    Returns
    -------
    tuple
        List of the validity of transitions in the cell.

    """
    bits = (cell_transition >> ((3 - orientation) * 4))
    return (bits >> 3) & 1, (bits >> 2) & 1, (bits >> 1) & 1, bits & 1


def set_transitions(cell_transition, orientation, new_transitions):
    """
    Set the possible transitions (e.g., (N,E,S,W), 4 elements tuple
    if no diagonal transitions allowed) available for an agent
    oriented in direction `orientation` and inside a cell with transitions
    `cell_transition'. A new `cell_transition` is returned with
    the specified bits replaced by `new_transitions`.

    Parameters
    ----------
    cell_transition : int
        16 bits used to encode the valid transitions for a cell.
    orientation : int
        Orientation of the agent inside the cell.
    new_transitions : tuple
        Tuple of new transitions validitiy for the cell.

    Returns
    -------
    int
        An updated bitmap that replaces the original transitions validity
        of `cell_transition' with `new_transitions`, for the appropriate
        `orientation`.

    """
    mask = (1 << ((4 - orientation) * 4)) - (1 << ((3 - orientation) * 4))
    negmask = ~mask

    new_transitions = \
        (new_transitions[0] & 1) << 3 | \
        (new_transitions[1] & 1) << 2 | \
        (new_transitions[2] & 1) << 1 | \
        (new_transitions[3] & 1)

    cell_transition = (cell_transition & negmask) | (new_transitions << ((3 - orientation) * 4))

    return cell_transition


def rotate_transition(cell_transition, rotation=0):
    """
    Clockwise-rotate a 16-bit transition bitmap by
    rotation={0, 90, 180, 270} degrees.

    Parameters
    ----------
    cell_transition : int
        16 bits used to encode the valid transitions for a cell.
    rotation : int
        Angle by which to clock-wise rotate the transition bits in
        `cell_transition` by. I.e., rotation={0, 90, 180, 270} degrees.

    Returns
    -------
    int
        An updated bitmap that replaces the original transitions bits
        with the equivalent bitmap after rotation.

    """
    # Rotate the individual bits in each block
    value = cell_transition
    rotation = rotation // 90
    for i in range(4):
        block_tuple = get_transitions(value, i)
        block_tuple = block_tuple[(4 - rotation):] + block_tuple[:(4 - rotation)]
        value = set_transitions(value, i, block_tuple)

    # Rotate the 4-bits blocks
    value = ((value & (2 ** (rotation * 4) - 1)) << ((4 - rotation) * 4)) | (value >> (rotation * 4))

    cell_transition = value
    return cell_transition


def load_pil_png(file_directory=None):
    if file_directory is None:
        file_directory = rail_files

    transition_image = {}

    directions = list("NESW")
    for transition, file_name in file_directory.items():
        transition_16_bit = ["0"] * 16
        for sTran in transition.split(" "):
            if len(sTran) == 2:
                in_direction = directions.index(sTran[0])
                out_direction = directions.index(sTran[1])
                transition_idx = 4 * in_direction + out_direction
                transition_16_bit[transition_idx] = "1"
        transition_16_bit_string = "".join(transition_16_bit)
        binary_trans = int(transition_16_bit_string, 2)

        img = Image.open("%s%s" % (image_dir, file_name))
        transition_image[binary_trans] = img
        for nRot in [90, 180, 270]:
            binary_trans_2 = rotate_transition(binary_trans, nRot)
            img_rotated = img.rotate(-nRot)
            transition_image[binary_trans_2] = img_rotated
    return transition_image


def load_png(file_directory=None):
    if file_directory is None:
        file_directory = target_files

    transition_image = {}

    directions = list("NESW")
    for transition, file_name in file_directory.items():
        transition_16_bit = ["0"] * 16
        for sTran in transition.split(" "):
            if len(sTran) == 2:
                in_direction = directions.index(sTran[0])
                out_direction = directions.index(sTran[1])
                transition_idx = 4 * in_direction + out_direction
                transition_16_bit[transition_idx] = "1"
        transition_16_bit_string = "".join(transition_16_bit)
        binary_trans = int(transition_16_bit_string, 2)

        transition_image[binary_trans] = {"src": "%s%s" % (image_dir, file_name), "rotation": 0}
    return transition_image


def load_png_rotation(file_directory=None):
    if file_directory is None:
        file_directory = rail_files

    transition_image = {}

    directions = list("NESW")
    for transition, file_name in file_directory.items():
        transition_16_bit = ["0"] * 16
        for sTran in transition.split(" "):
            if len(sTran) == 2:
                in_direction = directions.index(sTran[0])
                out_direction = directions.index(sTran[1])
                transition_idx = 4 * in_direction + out_direction
                transition_16_bit[transition_idx] = "1"
        transition_16_bit_string = "".join(transition_16_bit)
        binary_trans = int(transition_16_bit_string, 2)

        transition_image[binary_trans] = {"src": "%s%s" % (image_dir, file_name), "rotation": 0}
        for nRot in [90, 180, 270]:
            binary_trans_2 = rotate_transition(binary_trans, nRot)
            if binary_trans_2 not in transition_image:
                transition_image[binary_trans_2] = {"src": "%s%s" % (image_dir, file_name), "rotation": -nRot}
    return transition_image


def targets(level_data):
    target = []
    for agent in level_data["environmentData"]["agents"]:
        target.append(level_data["environmentData"]["grid"][agent["target"][0]][agent["target"][1]])
    return target


def load_target_all(level_data):
    transitions = load_png_rotation(rail_files)
    stations = load_png(target_files)
    for cell_type in targets(level_data):
        transitions[cell_type] = stations[cell_type]
    return transitions
