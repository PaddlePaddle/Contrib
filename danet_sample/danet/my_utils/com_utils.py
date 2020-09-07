import sys
def trainval_show(num, nums, acc, loss, prefix='', suffix=''):
    '''
    按进度条输出每个batch的acc、loss
    :param num: 当前batch数
    :param nums: 总batch数
    :param acc:
    :param loss:
    :param prefix:
    :param suffix:
    :return:
    '''
    rate = num / nums
    ratenum = round(rate, 3) * 100
    bar = '\r%s batch %3d/%d: acc %.4f, loss %00.4f [%s%s]%.1f%% %s; ' % (
        prefix, num, nums, acc, loss, '#' * (int(ratenum) // 2), '_' * (50 - int(ratenum) // 2), ratenum,
        suffix)
    sys.stdout.write(bar)
    sys.stdout.flush()

def process_show(num, nums, pre_fix='', suf_fix=''):
    rate = num / nums
    ratenum = round(rate, 3) * 100
    bar = '\r%s %g/%g [%s%s]%.1f%% %s' % \
          (pre_fix, num, nums, '#' * (int(ratenum) // 5), '_' * (20 - (int(ratenum) // 5)), ratenum, suf_fix)
    sys.stdout.write(bar)
    sys.stdout.flush()