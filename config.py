class BaseConfig:
    SECRET_KEY = "9KStWezC"


class LocalConfig(BaseConfig):
    DATA_FOLDER = "./stock_data/"
    TEMPLATES_FOLDER = "./templates/"


class ProductConfig(BaseConfig):
    DATA_FOLDER = "/var/www/python39/simulator_app/stock_data/"
    TEMPLATES_FOLDER = "/var/www/python39/simulator_app/templates/"


config = {
    "product": ProductConfig,
    "local": LocalConfig,
}
