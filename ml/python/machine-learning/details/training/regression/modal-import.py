import joblib


def modal_import():
    modal = joblib.load("./export/test.pkl")
    print(modal.coef_)
    return None


if __name__ == '__main__':
    modal_import()
