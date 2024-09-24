required_packages <- c("argparse", "magrittr", "dplyr", "mirt")

install_and_load <- function(packages) {
  for (package in packages) {
    if (!require(package, character.only = TRUE)) {
      install.packages(package, dependencies = TRUE)
      library(package, character.only = TRUE)
    }
  }
}

install_and_load(required_packages)