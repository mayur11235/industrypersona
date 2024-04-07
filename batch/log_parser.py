import datetime
import pandas as pd
import sys

def from_log(log_file,rating_file):
    user_id = 'default'
    rating = -1
    feedback = ''
    chat_data = []
    with open(rating_file, 'a') as rating_output:
        rating_output.write(f"userid,response\n")
    with open(log_file, 'r') as file:
        for line in file:
            log_type, *log_data = line.strip().split('|^')
            
            if log_type.endswith('UserRating'):
                user_id, rating = log_data
                with open(rating_file, 'a') as rating_output:
                    rating_output.write(f"{user_id},Rating: {rating}\n")
            elif log_type.endswith('UserFeedback'):
                user_id,feedback = log_data
                feedback=feedback.replace(',', ';').replace('\n', ' ')
                with open(rating_file, 'a') as rating_output:
                    rating_output.write(f"{user_id},Feedback: {feedback}\n")
            elif log_type.endswith('UserChat'):                    
                    user_id,record_index, *row_data = log_data
                    chat_data.append([user_id,int(record_index)] + row_data) 
    df = pd.DataFrame(chat_data, columns=['userid','index','role','content'])
    return df, user_id, int(rating), feedback

def save_as_csv(df, output_file):
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python log_parser.py <log_file>")
        sys.exit(1)
    log_file = sys.argv[1]
    current_timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    output_file = f"user_feedback_{current_timestamp}.csv"
    rating_file = f"user_rating_{current_timestamp}.csv"
    df, user_id, rating, feedback = from_log(log_file,rating_file)
    save_as_csv(df, output_file)
    