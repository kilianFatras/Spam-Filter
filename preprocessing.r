# ===========================================================
# ====================== Preprocessing ======================
# ===========================================================

set.seed(103)

# -------------------- Load the data --------------------

df = read.table("spam.data")
n = dim(df)[1]

# -------- Split into training and validation set--------

is_train = sample(c(TRUE,FALSE), n, rep = TRUE, prob=c(2/3,1/3))
