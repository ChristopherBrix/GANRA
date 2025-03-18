from pysmt.smtlib.parser import SmtLibParser


def get_formula(name):
    with open(name, "r", encoding="utf-8") as file:
        parser = SmtLibParser()
        script = parser.get_script(file)
        return script
