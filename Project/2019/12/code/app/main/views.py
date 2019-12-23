from app import get_logger, get_config
import math
from flask import render_template, redirect, url_for, flash, request
from flask_login import login_required, current_user
from app import utils
from app.models import CfgNotify, WB
from app.main.forms import CfgNotifyForm,SearchForm, SummaryForm
from . import main
import getcount
from wtforms import Form, TextAreaField, validators
import app.sentiment.predict as sent
import app.topic.lda as topic



logger = get_logger(__name__)
cfg = get_config()

# 通用列表查询
def common_list(DynamicModel, view):
    # 接收参数
    action = request.args.get('action')
    id = request.args.get('id')
    page = int(request.args.get('page')) if request.args.get('page') else 1
    length = int(request.args.get('length')) if request.args.get('length') else cfg.ITEMS_PER_PAGE

    # 删除操作
    if action == 'del' and id:
        try:
            DynamicModel.get(DynamicModel.id == id).delete_instance()
            flash('删除成功')
        except:
            flash('删除失败')

    # 查询列表
    query = DynamicModel.select()
    total_count = query.count()
    length = 30
    # 处理分页
    if page: query = query.paginate(page, length)

    dict = {'content': utils.query_to_list(query), 'total_count': total_count,
            'total_page': math.ceil(total_count / length), 'page': page, 'length': length}
    return render_template(view, form=dict, current_user=current_user)


# 通用单模型查询&新增&修改
def common_edit(DynamicModel, form, view):
    id = request.args.get('id', '')
    if id:
        # 查询
        model = DynamicModel.get(DynamicModel.id == id)
        if request.method == 'GET':
            utils.model_to_form(model, form)
        # 修改
        if request.method == 'POST':
            if form.validate_on_submit():
                utils.form_to_model(form, model)
                model.save()
                flash('修改成功')
            else:
                utils.flash_errors(form)
    else:
        # 新增
        if form.validate_on_submit():
            model = DynamicModel()
            utils.form_to_model(form, model)
            model.save()
            flash('保存成功')
        else:
            utils.flash_errors(form)
    return render_template(view, form=form, current_user=current_user, count=getcount.getStasticInfo())



# 情感查询
def sentiment_analysis(form, view):

    return render_template(view,form=form,current_user=current_user)
class ReviewForm(Form):
    sentimentreview = TextAreaField('',
                                [validators.DataRequired(),
                                validators.length(min=15)])
# 根目录跳转
@main.route('/', methods=['GET'])
def root():

    return redirect(url_for('main.index'))


# 首页
@main.route('/index', methods=['GET'])
def index():
    topics,num = getcount.getTopics()
    return render_template('index.html', current_user=current_user, count=getcount.getStasticInfo(),topics=topics,num=num)


# 通知方式查询
@main.route('/notifylist', methods=['GET', 'POST'])
def notifylist():
    return common_list(CfgNotify, 'notifylist.html')


# 通知方式配置
@main.route('/notifyedit', methods=['GET', 'POST'])
def notifyedit():
    return common_edit(CfgNotify, CfgNotifyForm(), 'notifyedit.html')

@main.route('/datainfo',methods=['GET', 'POST'])
def datainfo():
    return common_list(WB, 'datainfo.html')

# 查询页面
@main.route('/search', methods=['GET','POST'])
def search():
    form = SearchForm(request.form)
    if request.method == 'POST':
        review = request.form['sentimentReview']
        positive_prob, confidence, negative_prob, sentiment = sent.sent_predict(review)
        return render_template('results.html', current_user=current_user, content=review, sentiment=sentiment,
                               positive_prob=positive_prob,
                               negative_prob=negative_prob, confidence=confidence)
    return render_template('query.html', current_user=current_user, form=form)


# summary页面
@main.route('/summary', methods=['GET','POST'])
def summary():
    form = SummaryForm(request.form)
    if request.method == 'POST':
        title = request.form['texttitle']
        content = request.form['textcontent']
        LDA = topic.LDAClustering()
        tag, score = LDA.lda_predict(title, content, stopwords_path='./stop_words.txt', max_iter=100, n_components=30)
        print(tag, score)
        return render_template('summaryres.html',current_user=current_user, title=title, content=content,
                               tag=tag, score=score)
    return render_template('lda.html', current_user=current_user, form=form)


@main.route('/analysis', methods=['GET','POST'])
def analysis():
    id = request.args.get('id', '')
    label = getcount.getLabel(int(id))
    content, sentiment, positive_prob, negative_prob, confidence = getcount.getInfoByID(int(id))
    return render_template('analysis.html', current_user=current_user, content=content,sentiment=sentiment,positive_prob=positive_prob,
                           negative_prob=negative_prob, confidence=confidence,label=label)

