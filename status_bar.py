import sys


bar_length = 50
fill_character = '█'
empty_character = ' '


def get_text_by_phase(phase: int):
    """
    Generates a string based on a phase integer.
    :param phase: Integer denoting the phase.
    :return: Phase description as a string.
    """
    if phase == 0:
        return 'select parameters k and alpha'
    if phase == 1:
        return 'pre-process dataset'
    if phase == 2:
        return 'running anomaly detector'
    if phase == 3:
        return 'evaluating results'
    if phase == 4:
        return 'calculating objective function value'


def draw_status_bar(progress: float, maximum: float, text: str = '', phase: int = -1, on: bool = True):
    """
    Draws a status bar that shows progress compared to the maximum value.
    :param progress: The part already done out of the maximum possible.
    :param maximum: Maximum possible score.
    :param text: Text to display next to the status bar.
    :param phase: An integer showing which stage of some process the system is currently at.
    If not -1, it replaces the text based on the get_text_by_phase() function.
    :param on: whether to draw status bar
    :return: No return value.
    """
    if not on:
        return

    if phase != -1:
        text = get_text_by_phase(phase)

    percentage = round(100 * progress / maximum, 2)
    block_number = int(round(progress / maximum * bar_length))
    to_print = '\r|{0}| {1}/{2} ({3}%) {4}'.format(block_number * fill_character +
                                                   empty_character * (bar_length - block_number),
                                                   progress, maximum, percentage, text)
    to_print += (100-len(to_print))*' '
    sys.stdout.write(to_print)
    sys.stdout.flush()
    return


def draw_batch_status_bar(progress_for_executions, executions_max, run_for: int = 1):
    """
    Draws status bars that show progress compared to the maximum values.
    This function can display multiple status bars below each other.
    :param progress_for_executions: A dictionary showing how each process is progressing.
    :param executions_max: A dictionary that shows maximum values dor all processes.
    :param run_for: An integer showing how many iterations the system expects to perform.
    :return: No return value.
    """
    print('-' * 30)
    for key, value in progress_for_executions.items():
        percentage = round(100 * (value + 1) / executions_max[key], 2)
        block_number = int(round((value + 1) / executions_max[key] * bar_length))

        if run_for == 1:
            print("TS = {0:+.2f}:\t |{1}| {2}/{3} ({4} %)".format(
                key+1, block_number * fill_character + empty_character * (bar_length - block_number),
                value+1, executions_max[key], percentage))
        else:
            print("Execution {0:02d}/{1}:\t |{2}| {3}/{4} ({5} %)".format(
                key+1, run_for, block_number * fill_character + empty_character * (bar_length - block_number),
                value+1, executions_max[key], percentage))
    print('-' * 30)
    return
