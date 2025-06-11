
setwd('C:/Users/User/Desktop/Alina-SpikeSorting')

# Erforderliche Pakete laden
library(lme4)
library(glmmTMB)
library(ggpubr)
library(broom)
library(ggplot2)

# CSV-Datei einlesen
df <- read.csv("all_collected_scores_spike.csv")
df$Model <- as.factor(df$Model)
df$FeatureSet <- as.factor(df$FeatureSet)
df$Model <- relevel(df$Model, ref = "spike")
df$FeatureSet <- relevel(df$FeatureSet, ref = "spike")

# Liste der Spaltennamen (Metriken)
columns <- list("TP", "FP", "FN")

# Schleife über die angegebenen Spaltennamen
for (c in columns) {
  
  # Poisson-Modell erstellen
  model_formula <- as.formula(paste(c, "~ Model + offset(log(RecordingDuration)) + (1 | Dataset)"))
  model_poisson <- glmer(model_formula, family = poisson(link = "log"), data = df)
  
  # Negative Binomial-Modell erstellen
  model_nb <- glmmTMB(model_formula, family = nbinom2(), data = df)
  
  # AIC-Werte vergleichen
  aic_poisson <- AIC(model_poisson)
  aic_nb <- AIC(model_nb)
  print(paste("AIC Poisson for", c, ":", aic_poisson))
  print(paste("AIC Negative Binomial for", c, ":", aic_nb))
  
  # Überprüfung auf Überdispersion im Poisson-Modell
  pearson_resid_poisson <- residuals(model_poisson, type = "pearson")
  dispersion_ratio_poisson <- sum(pearson_resid_poisson^2) / df.residual(model_poisson)
  print(paste("Dispersion Ratio Poisson for", c, ":", dispersion_ratio_poisson))
  
  # Überprüfung auf Überdispersion im negativen Binomial-Modell
  pearson_resid_nb <- residuals(model_nb, type = "pearson")
  dispersion_ratio_nb <- sum(pearson_resid_nb^2) / df.residual(model_nb)
  print(paste("Dispersion Ratio Negative Binomial for", c, ":", dispersion_ratio_nb))
  
  
  sink(paste0("SummaryModel_", c, ".txt"))
  print(summary(model_nb))
  sink()
  
  # Modell zusammenfassen
  model_summary <- summary(model_nb)
  
  # Extrahiere feste Effekte
  fixed_effects <- model_summary$coefficients$cond
  
  # Erstelle einen DataFrame für ggplot2
  effects_df <- data.frame(
    term = rownames(fixed_effects),
    estimate = fixed_effects[, "Estimate"],
    std.error = fixed_effects[, "Std. Error"]
  )
  
  # Plot der Effektgrößen mit Konfidenzintervallen
  plot<-ggplot(effects_df, aes(x = term, y = estimate)) +
    geom_point() +
    geom_errorbar(aes(ymin = estimate - std.error * 1.96,
                      ymax = estimate + std.error * 1.96), width=0) +
    labs(title="Effect Sizes with Confidence Intervals") +
    theme(axis.text.x = element_text(angle=45, hjust=1))
  jpeg(file=paste0("NBPlotEffekte_", c, ".jpeg"))
  print(plot)
  dev.off()
  
  # Visualisierung der Vorhersagen für eine Metrik
  predictions <- predict(model_nb, type = "response")
  df$predictions <- predictions
  
  
  plot<-ggplot(df, aes(x = Model, y = predictions)) +
    geom_boxplot() +
    facet_wrap(~ FeatureSet) +
    labs(title = paste("Predictions for", c))
  jpeg(file=paste0("NBPlot_", c, ".jpeg"))
  print(plot)
  dev.off()
}
