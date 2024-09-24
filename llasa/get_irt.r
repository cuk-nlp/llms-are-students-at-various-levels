library(magrittr)
library(dplyr)
library(mirt)
library(argparse)

# Set up the argument parser
parser <- ArgumentParser()
parser$add_argument("-o", "--output_path", help="Path to output csv file")
parser$add_argument("-d", "--data_type", help="Data type")
parser$add_argument("-t", "--train_type", help="Train type")
args <- parser$parse_args()

# Load and preprocess the dataset
df <- read.csv(paste(args$output_path, "/", args$data_type, "_", args$train_type, ".csv", sep=""))
df <- df %>% arrange(question_id)

# Transpose the dataset for IRT analysis
transposed_data <- t(df)
colnames(transposed_data) <- transposed_data["question_id", ]  # Set question_id as column names
transposed_data <- transposed_data[!(rownames(transposed_data) == 'question_id'), ]  # Remove 'question_id' row

# Convert to numeric and filter rows with all 0s or all 1s
transposed_data <- as.data.frame(transposed_data)
transposed_data[] <- lapply(transposed_data, as.numeric)

# Identify columns with all 0s or all 1s
all_zeros <- apply(transposed_data, 2, function(x) all(x == 0))
all_ones <- apply(transposed_data, 2, function(x) all(x == 1))

# Subset data excluding all 0s and all 1s for IRT calculation
filtered_data <- transposed_data[, !(all_zeros | all_ones)]

# IRT analysis on filtered data
irt_data <- mirt(filtered_data, 1, itemtype = 'Rasch', technical = list(NCYCLES = 2000))
irt_coef <- coef(irt_data, IRTpars = TRUE, simplify = TRUE)
irt_difficulty <- irt_coef$items[, 'b']
irt_ability <- fscores(irt_data)

# Adjust difficulty for all 0s and all 1s columns
max_difficulty <- max(irt_difficulty)
min_difficulty <- min(irt_difficulty)

# Prepare full difficulty vector with question_id as names
irt_difficulty_full <- rep(NA, ncol(transposed_data))
names(irt_difficulty_full) <- colnames(transposed_data)  # Set question IDs as names

# For all 0s, set to max difficulty
irt_difficulty_full[all_zeros] <- max_difficulty

# For all 1s, set to min difficulty
irt_difficulty_full[all_ones] <- min_difficulty

# Merge IRT difficulties back
irt_difficulty_full[!(all_zeros | all_ones)] <- irt_difficulty

# Print the difficulty vector (optional)
# print(irt_difficulty_full)

# Write the results to CSV files, including row names (question_ids)
write.csv(irt_difficulty_full, paste(args$output_path, "/", args$data_type, "_", args$train_type, "_difficulty.csv", sep=""), row.names=TRUE)
write.csv(irt_ability, paste(args$output_path, "/", args$data_type, "_", args$train_type, "_ability.csv", sep=""))
