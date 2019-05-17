#install.packages("titanic")
library(titanic)
#install.packages("ggplot2")
library(ggplot2)
#install.packages("caret")
library(caret)

dane<-titanic_train
dim(dane) #sprawdzenie rozmiaru danych

head(dane,10) #sprawdzam jak wygladaja moje dane

#Usuwam zbedne dane (nie wnoszace nic do przezywalnosci)
do_usuniecia <- which(colnames(dane) %in% c("PassengerId","Name","Ticket","Cabin"))

dane <- dane[,-do_usuniecia]

#sprawdzenie typow danych
str(dane)

#zmiana typow na factory

library(dplyr)
dane <- dane %>%
  mutate_at(
    .vars = vars("Survived","Pclass","Sex","Embarked"),
    .funs = funs(as.factor(.))
  )

str(dane)

#Analiza brakow
sapply(dane, function(x) sum(is.na(x))) #177 braki w Age, zajme sie nimi pozniej
sapply(dane, function(x) length(which(x==" "))) #brak
sapply(dane, function(x) length(which(x=="?"))) #brak
sapply(dane, function(x) length(which(x==""))) # 2 braki w Embarked

#Usuniecie 2 brakow z Embarked
v <- which(dane$Embarked=="")
dane <- dane[-v,]
sapply(dane, function(x) length(which(x==""))) #brak

#Pierwsze podsumowanie danych
summary(dane)

#usuniecie pustego levelu z Embarked
dane$Embarked <- factor(dane$Embarked)
summary(dane)

#Zmiana poziomow
levels(dane$Pclass) <- c("Upper", "Middle", "Lower")

### WARTOSCI ODSTAJACE ###

summary(dane) #dane odstajace moga byc w Age i Fare (duza dysproorcja miedzy
# srednia i mediana a wartoscia najwieksza)

#Pierwszy sposob na dane odstajace - wykres punktowy
ggplot(dane)+
  aes(x=Age, y=Age)+
  geom_point()
# 5 wartosci odstajacych

ggplot(dane)+
  aes(x=Fare,y=Fare)+
  geom_point()
# sporo wartosci odstajacych

#Drugi sposob na dane odstajace - wykres pudelkowy
boxplot(dane$Age)
boxplot(dane$Fare)

#Trzeci sposob na dane odstajace - 

outlr = function(x.na){
  x <- na.omit(x.na)
  lower.boundry <- quantile(x, 0.25) - IQR(x) * 1.5
  upper.boundry <- quantile(x, 0.75) + IQR(x) * 1.5
  num.of.outliers.u <- sum(x>upper.boundry)
  num.of.outliers.l <- sum(x<lower.boundry)
  procent <- round((num.of.outliers.l+num.of.outliers.u)/
                     length(x.na)*100,2)
  return(data.frame(lower.boundry, upper.boundry, 
                    num.of.outliers.l, num.of.outliers.u,procent))
}
outlr(dane$Age) # wyszlo 8 wartosci odstajacych czyli 0.9% wszystkich 
#gorna granica wyszla 65
outlr(dane$Fare) # wyszlo 114 wartosci odstajacych czyli ok 13% wszystkich
#gorna granica wyszla 65.6563

#Zastapienie wartosci odstajacych mediana
odst_1 <- which(dane$Age>65.00)
dane$Age[odst_1] <- mean(dane$Age,na.rm = T)
odst_2 <- which(dane$Fare>65.6563)
dane$Fare[odst_2] <- mean(dane$Fare,na.rm = T)

#sprawdzenie
summary(dane)

ggplot(dane)+
  aes(x=Age, y=Age)+
  geom_point()

ggplot(dane)+
  aes(x=Fare,y=Fare)+
  geom_point()

## Po uporaniu sie z wartosciami odstajacymi czas na missing value

# W kolumnie Age byly missing value
sum(is.na(dane$Age))

dane$Age[is.na(dane$Age)] <- median(dane$Age,na.rm = T)

#Anomalie w zmiennych kategorycznych
summary(dane) #brak anomialii - pojedynczych obserwacji dla danej kategorii


## Rozpoznanie rozkladu zmiennych numerycznych
## celem doboru odpowiedniej metody korelacji

# Sposob 1 za pomoca skosnosci rozkladu
dane.numeric <- dane[,sapply(dane,function(x) is.numeric(x))]
skosnosc <- sapply(dane.numeric,
                   function(x) e1071::skewness(x,na.rm = T))
skosnosc
#Wnioski:
#wszystkie sa prawostronnie skosne
# mozliwe, ze nie pochodza z rozkladu normalnego

# Sposob 2 za pomoca histogramow
sapply(dane.numeric,hist)
#Wnioski : Age moze byc z rozkladu normalnego, reszta nie

# Sposob 3 - za pomoca testu Shiapiro-Wilka
sapply(dane.numeric, function(x) shapiro.test(x))

## Wnioski - brak wskazan do uzycia korelacji Pearsona

# Pozbywam sie zmiennych skorelowanych na poziomie >=75%
abs(cor(dane.numeric,method = "spearman"))
#brak zmiennych numerycznych skorelowanych na takim poziomie

## Przeksztalcenie zmiennych kategorycznych na wektory
surv <- which(colnames(dane) %in% c("Survived"))
dane01 <- mlr::createDummyFeatures(dane[,-surv])
dane01$Survived<-dane$Survived
head(dane01)

#Podzial na zbior testowy i treningowy
set.seed(16052019)
wiersze <- createDataPartition(dane01$Survived, p=0.8, list = F)
zbior_treningowy <- dane01[wiersze,]
zbior_testowy <- dane01[-wiersze,]

#sprawdzenie podzialu zbiorow
dim(zbior_treningowy)
prop.table(table(zbior_treningowy$Survived))
dim(zbior_testowy)
prop.table(table(zbior_testowy$Survived))
#proporcje zmiennej celu sa zachowane

#ustawienia walidacji krzyzowej
control <- trainControl(method="repeatedcv", number=10, repeats = 3)

#Budowa pelnego modelu
model1 <- train(Survived~., data = zbior_treningowy, 
                         method="glm",
                         trControl = control)

acc_tren1<-round(max(model1$results$Accuracy),3)
acc_tren1 #0.787
model1$bestTune #w tym algorytmie nie ma hiperparametrow

przewidywanie_model1 <- predict(model1,zbior_testowy)
l <- length(zbior_testowy$Survived)
acc_test1 <- round(sum(przewidywanie_model1==zbior_testowy$Survived)/l,3)
acc_test1 #0.797

#moim wyjsciowym accuracy bedzie 79.7% wszystko co bedzie wieksze badz rowne
#uznam za sukcesi zakoncze projekt

#model2 - wybor zmiennych poprzez RFE
control_rfe <- rfeControl(functions=rfFuncs, method="cv", number=10)
# run the RFE algorithm
results <- rfe(Survived~.,data = zbior_treningowy, rfeControl=control_rfe)
#rezultaty
plot(results, type=c("g", "o"))
results$variables

#dobre dopasowanie dla 8 zmiennych:
#Parch,Pclass.Middle,SibSp,Age,Fare,Pclass.Lower
#Sex.female,Sex.male

model2 <- train(Survived~Parch+Pclass.Middle+SibSp+Age+
                  Fare+Pclass.Lower+Sex.female+Sex.male,
                data = zbior_treningowy,
                method = "glm",
                trControl = control)

#zbior_treningowy
model2$coefnames

acc_tren2<-round(max(model2$results$Accuracy),3)
acc_tren2 #0.787

przewidywanie_model2 <- predict(model2,zbior_testowy)
l <- length(zbior_testowy$Survived)
acc_test2 <- round(sum(przewidywanie_model2==zbior_testowy$Survived)/l,3)
acc_test2 #0.78

#Wnioski: Lepze dopasowanie mamy dla pelnego zestawu zmiennych