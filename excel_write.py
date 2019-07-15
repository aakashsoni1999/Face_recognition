import xlwt
from xlwt import Workbook
import datetime
from datetime import datetime

wb = Workbook()

sheet = wb.add_sheet('Log Record')

sheet.write(0, 0, 'Sno.')
sheet.write(0, 1, 'Name')
sheet.write(0, 2, 'Status')
sheet.write(0, 3, 'Time')
sheet.write(0, 4, 'Date')
sheet.write(0, 5, 'Total_time')
sheet.write(0, 6, 'Hours')
sheet.write(0, 7, 'Minutes')
sheet.write(0, 8, 'Seconds')

def write_in_file(current,previous,total_time,persons_time,s_no):
    for i in current:
        flag = 0
        for j in previous:
            if(i == j):
                flag = 1
                break
        if(flag == 0):
            sheet.write(s_no, 0, str(s_no))
            sheet.write(s_no, 1, str(i))
            sheet.write(s_no, 2, 'ENTRY')
            sheet.write(s_no, 3, str(datetime.now().time()))
            sheet.write(s_no, 4, str(datetime.now().date()))
            if str(i) in persons_time:
                time_diff = datetime.now() - persons_time[str(i)]
                t_seconds = time_diff.total_seconds()
                hours = t_seconds // 3600
                sheet.write(s_no, 6, str(hours))
                t_seconds = t_seconds % 3600
                minutes = t_seconds // 60
                sheet.write(s_no, 7, str(minutes))
                t_seconds = t_seconds % 60
                seconds = t_seconds
                sheet.write(s_no, 8, str(seconds))
                if str(i) in total_time:
                    total_time[str(i)] = total_time[str(i)] + t_seconds
                else:
                    total_time[str(i)] = t_seconds
            s_no = s_no + 1

    for i in previous:
        flag = 0
        for j in current:
            if(i == j):
                flag = 1
                break
        if(flag == 0):
            sheet.write(s_no, 0, str(s_no))
            sheet.write(s_no, 1, str(i))
            sheet.write(s_no, 2, 'EXIT')
            sheet.write(s_no, 3, str(datetime.now().time()))
            sheet.write(s_no, 4, str(datetime.now().date()))
            if str(i) in persons_time:
                del persons_time[str(i)]
            else:
                persons_time[str(i)] = datetime.now()
            s_no = s_no + 1
    return s_no

def final_write(total_time,s_no):
    for i in total_time:
        sheet.write(s_no, 0, str(i))
        sheet.write(s_no, 2, total_time[i])
        s_no = s_no + 1
    wb.save('Log record.ods')
