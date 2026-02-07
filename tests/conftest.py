def pytest_report_header(config):
    headers = []
    try:
        from pywr.model import Model

        solver_name = Model().solver.name
    except Exception as e:
        # Allow running pure-Python tests in environments where the C-extensions
        # are not built (e.g. missing Python dev headers).
        solver_name = f"unavailable ({type(e).__name__})"
    headers.append("solver: {}".format(solver_name))
    return "\n".join(headers)
