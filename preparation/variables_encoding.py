class Encoding:

    def __init__(self, filename, file_tag, data):
        self.filename = filename
        self.file_tag = file_tag
        self.data     = data

    def encode_credit(self):
        encoded_data = self.data.copy().dropna()
        #encoded_data = self.data.copy()

        encoding = {
            # Symbolic (Ordinal)
            "CreditMix": {"Bad": 0, "Standard": 1, "Good": 2},
            "Payment_Behaviour": {"Low_spent_Small_value_payments": 0, "Low_spent_Medium_value_payments": 1,
                                  "Low_spent_Large_value_payments": 2, "High_spent_Small_value_payments": 3,
                                  "High_spent_Medium_value_payments": 4, "High_spent_Large_value_payments": 5},
            "Month": {"January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
                      "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12},

            # Binary - mudar c a da prof so yes no Yes No wtv
            "Payment_of_Min_Amount": {"No": 0, "Yes": 1},
            "Credit_Score": {"Poor": 0, "Good": 1},
        }

        # Encoding ordinal variables
        encoded_data.replace(encoding, inplace=True)

        # Dummification (One-hot encoding) (non-ordenal) ???????????????
        # The dummifying 'State' will create separate binary columns for each unique state, increasing the dimensionality of your dataset. - help
        # vars: list[str] = ["Occupation", "Type_of_Loan"]
        # df_encoded = dummify(encoded_data, vars)

        # return df_encoded
        return encoded_data