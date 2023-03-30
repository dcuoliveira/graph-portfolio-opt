from models.GCLSTM import GCLSTM
import mgarch

DCC_GARCH = mgarch.mgarch()

model_names = {
    "gclstm": GCLSTM,
    "dcc_garch": DCC_GARCH
}