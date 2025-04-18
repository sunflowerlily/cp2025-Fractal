import turtle
import time

def apply_rules(axiom, rules, iterations):
    """
    Applies the L-system production rules for a given number of iterations.
    Args:
        axiom (str): The starting string (axiom).
        rules (dict): A dictionary where keys are symbols to replace
                      and values are the replacement strings.
        iterations (int): The number of times to apply the rules.
    Returns:
        str: The final string after all iterations.
    """
    current_string = axiom
    print(f"Axiom: {axiom}")
    for i in range(iterations):
        next_string = ""
        for char in current_string:
            # Replace char if it's in rules, otherwise keep it
            next_string += rules.get(char, char)
        current_string = next_string
        # Optional: Print intermediate strings for debugging/understanding
        # print(f"Iteration {i+1} (length {len(current_string)}): {current_string[:100]}...") # Print first 100 chars
    print(f"Finished generating instructions (length {len(current_string)}).")
    return current_string

def draw_l_system(turtle_obj, instructions, angle, step_length):
    """
    Interprets the L-system string and draws it using Turtle graphics.
    Args:
        turtle_obj (turtle.Turtle): The turtle object to draw with.
        instructions (str): The final L-system string.
        angle (float): The angle to turn for '+' and '-'.
        step_length (float): The distance to move forward for 'F' or 'G'.
    """
    stack = [] # To store position and heading for '[' and ']'

    print("Starting drawing...")
    start_time = time.time()

    for command in instructions:
        if command == 'F' or command == 'G': # Treat F and G as "draw forward"
            turtle_obj.forward(step_length)
        elif command == '+': # Turn left
            turtle_obj.left(angle)
        elif command == '-': # Turn right
            turtle_obj.right(angle)
        elif command == '[': # Push state: position and heading
            stack.append((turtle_obj.position(), turtle_obj.heading()))
        elif command == ']': # Pop state: restore position and heading
            position, heading = stack.pop()
            turtle_obj.penup()
            turtle_obj.goto(position)
            turtle_obj.setheading(heading)
            turtle_obj.pendown()
        # Ignore other characters (like X, Y if used as variables)

    end_time = time.time()
    print(f"Drawing complete in {end_time - start_time:.2f} seconds.")

# --- L-System Parameters for a Ginkgo-like Shape ---
# These parameters are experimental and simplified.

# Parameters Set 1: Simple Fan/Bush (adjust angle/iterations)
# AXIOM = "X"
# RULES = {
#     "X": "F-[[X]+X]+F[+FX]-X", # A common plant-like rule
#     "F": "FF"                  # Makes lines grow longer
# }
# ITERATIONS = 4             # Fewer iterations due to F->FF growth
# ANGLE = 25                 # Branching angle
# STEP = 5                   # Initial step length

# Parameters Set 2: Attempt at Wider Fan (More Ginkgo-like)
AXIOM = "Y" # Start with a symbol representing the growing tip
RULES = {
    "Y": "F[+Y][-Y]FY", # Branch left, branch right, draw stem, continue tip
    "F": "G",           # Optional: Change F to G to stop F from re-applying rule Y if Y->F...
    "G": "G"            # G just draws (or could grow: G->GG)
    # "F": "FF" # Alternative: Make segments grow longer
}
ITERATIONS = 6             # More iterations for detail
ANGLE = 30                 # Wider angle for fan shape
STEP = 4                   # Step length

# --- Turtle Setup ---
screen = turtle.Screen()
screen.setup(width=800, height=900)
screen.bgcolor("white")
screen.title("L-System Ginkgo Leaf Fractal")

# Create the turtle
pen = turtle.Turtle()
pen.speed(0)           # Fastest speed
pen.hideturtle()       # Hide the turtle arrow
pen.penup()
# Starting position: bottom center, pointing up
pen.goto(0, -screen.window_height() / 2 + 50)
pen.left(90)          # Point upwards
pen.pendown()
pen.color("darkgreen") # Ginkgo-like color

# Optimize drawing speed
screen.tracer(0) # Turn off screen updates during drawing

# --- Generate and Draw ---
instructions = apply_rules(AXIOM, RULES, ITERATIONS)
draw_l_system(pen, instructions, ANGLE, STEP)

# Finish drawing
screen.update() # Show the final drawing
print("Drawing finished. Close the window to exit.")
screen.mainloop() # Keep the window open until closed