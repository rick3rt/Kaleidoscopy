# def create_line(point, angle_degrees):
#     x0, y0 = point
#     angle_radians = np.radians(angle_degrees)
#     # Handle vertical lines (slope is undefined)
#     if angle_degrees % 180 == 90:
#         def vertical_line(x):
#             x = np.asarray(x)
#             if np.any(x != x0):
#                 raise ValueError(
#                     f"Vertical line at x = {x0}. y is undefined for x ≠ {x0}."
#                 )
#             return np.full_like(x, y0)
#         return vertical_line
#     # Calculate the slope
#     slope = np.tan(angle_radians)
#     # Return the line function: y = m*(x - x0) + y0
#     def line_function(x):
#         x = np.asarray(x)
#         return slope * (x - x0) + y0
#     return line_function


def create_line(intercept, direction):

    x0, y0 = intercept
    dx, dy = direction

    # Handle vertical lines where dx = 0
    if np.isclose(dx, 0):

        def vertical_line(x):
            x = np.asarray(x)
            if np.any(~np.isclose(x, x0)):
                raise ValueError(
                    f"Vertical line at x = {x0}. y is undefined for x ≠ {x0}."
                )
            return np.full_like(x, y0)

        # Parametric form for vertical line
        def parametric_line(t):
            return np.full_like(t, x0), y0 + dy * t

        return vertical_line, parametric_line

    # Calculate slope
    slope = dy / dx

    # Slope-intercept form: y = m*(x - x0) + y0
    def line_function(x):
        x = np.asarray(x)
        return slope * (x - x0) + y0

    # Parametric form: (x, y) = (x0 + dx * t, y0 + dy * t)
    def parametric_line(t):
        t = np.asarray(t)
        return x0 + dx * t, y0 + dy * t

    return line_function, parametric_line
