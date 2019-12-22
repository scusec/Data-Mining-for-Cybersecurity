$(document).ready(function(){
  'use strict';

  /* SETUP VARS */
  var winWidth = $(window).width();
  var winHeight = $(window).height();

  /* LIGHBOX */
  $('.swipebox').swipebox({
    hideBarsDelay: 99999999,
    useSVG: false
  });

  // ANIMATE ONCE PAGE LOAD IS OVER //
  Pace.on("done", function(){
      new WOW().init();
  });

  //caches a jQuery object containing the header element
  var header = $("#main-navigation");
  $(window).scroll(function() {
      var scroll = $(window).scrollTop();

      if (scroll >= 250) {
          header.removeClass('opaqued').addClass("opaqued-dark");
          $('#searchwrapper').addClass("opaqued-dark");
      } else {
          header.removeClass("opaqued-dark").addClass('opaqued');
          $('#searchwrapper').removeClass("opaqued-dark");
      }
  });

  //fullscreen
  if(winWidth >= 768){
    $('.fullscreen').css({
      'height': winHeight,
    });
  }
  $(window).resize(function(){
    if(winWidth >= 768){
      $('.fullscreen').css({
        'height': winHeight,
      });
    }
  });

  $(window).on('scroll', function () {
      var st = $(this).scrollTop();
      $('.post-heading').each(function () {
          var offset = $(this).offset().top;
          var height = $(this).outerHeight();
          offset = offset + height / 2;
          $(this).css({ 'opacity': 1 - (st - offset + 200) / 200 });
      });
  });

  //JRIBBBLE
  $.jribbble.getShotsByPlayerId('pletikos', function (playerShots) {
    var html = [];
    $.each(playerShots.shots, function (i, shot) {
      html.push('<li class="col-xs-4"><figure class="hover-item">');  
        html.push('<img class="img-responsive" src="' + shot.image_teaser_url + '" ');
        html.push('alt="' + shot.title + '">');
        html.push('<figcaption>');
        html.push('<h2>' + shot.title + '</h2>');
        html.push('<p class="icon-links"><a href="' + shot.url + '">');
        html.push('<span class="fa fa-info"></span></a></p>');
        html.push('</figcaption>');
      html.push('</figure></li>');
    });

    $('#jribble-list').html(html.join(''));
  }, {page: 1, per_page: 6});

  //CONTACT FORM
  $('#contactform').submit(function(){
    var action = $(this).attr('action');
    $("#message").slideUp(750,function() {
    $('#message').hide();
    $('#submit').attr('disabled','disabled');
    $.post(action, {
      name: $('#name').val(),
      email: $('#email').val(),
      website: $('#website').val(),
      comments: $('#comments').val()
    },
      function(data){
        document.getElementById('message').innerHTML = data;
        $('#message').slideDown('slow');
        $('#submit').removeAttr('disabled');
        if(data.match('success') != null) $('#contactform').slideUp('slow');
        $(window).trigger('resize');
      }
    );
    });
    return false;
  });

  $('.owl-carousel').owlCarousel({
    navigation: true,
    pagination: false,
    navigationText: [
    "<i class='fa fa-angle-left'></i>",
    "<i class='fa fa-angle-right'></i>"
    ], 
    autoPlay: 8000
  });

  $('.owl-carousel-paged').owlCarousel({
    navigation: false,
    pagination: true,
    autoPlay: 8000
  });

  // MASONRY
  var $container = $('#masonry-blog');
  // initialize
  $container.imagesLoaded( function() {
    $container.masonry({
      itemSelector: '.masonry-blog-item',
      columnWidth: $container.width() / 3
    });    
  });

  // SEARCH
  $('a#searchtrigger').click(function(e){
    e.preventDefault();
    var checkSearch = $('#searchwrapper').is(":hidden");

    if(checkSearch == true) {  
      $(this).find('.fa').removeClass('fa-search');
      $('#searchwrapper').slideDown(500);
      $('#searchwrapper input').fadeIn(1000);
      $('#main-navigation').addClass('search-open');
      $(this).find('.fa').addClass('fa-times');
    } else {
      $(this).find('.fa').removeClass('fa-times');
      $('#searchwrapper input').fadeOut(250);
      $('#searchwrapper').slideUp(500);
      $('#main-navigation').removeClass('search-open');
      $(this).find('.fa').addClass('fa-search');  
    }
  });

  // ONEPAGER //
  $('a.page-scroll').bind('click', function(event) {
      var $anchor = $(this);
      $('html, body').stop().animate({
          scrollTop: $($anchor.attr('href')).offset().top
      }, 1500, 'easeInOutExpo');
      event.preventDefault();
  });

  //NEAT VID EMBEDS
  $().prettyEmbed({ useFitVids: true });
  $(".container").fitVids();

  //COUNTER
  jQuery('.countup').counterUp({
      delay: 10,
      time: 1000
  });

  $('.mh').matchHeight();

  // Add slideDown animation to dropdown
  $('.dropdown').on('show.bs.dropdown', function(e){
    $(this).find('.dropdown-menu').first().stop(true, true).slideDown();
  });

  // Add slideUp animation to dropdown
  $('.dropdown').on('hide.bs.dropdown', function(e){
    $(this).find('.dropdown-menu').first().stop(true, true).slideUp();
  });

  $('[data-toggle="tooltip"]').tooltip()

  //SIDE MENU
  var mySlidebars = new $.slidebars();
  $('#launch-menu').on('click', function(e) {
      e.preventDefault();
      mySlidebars.slidebars.toggle('right');
  });

  $('#trigger-overlay').on('click', function(e) {
    e.preventDefault();
  });  

  $("#side-menu").metisMenu();
});

$(window).load(function() {

    jQuery('.preloader').fadeIn(300);
    setTimeout(function(){
      window.location = href;
    }, 650);
    return false;

});

/*-----------------------------------------------------------------------------------*/
/*  PORTFOLIO
/*-----------------------------------------------------------------------------------*/
(function($, window, document, undefined) {
"use strict";

    var gridContainer = $('#grid-container'),
        filtersContainer = $('#filters-container'),
        wrap, filtersCallback;


    /*********************************
        init cubeportfolio
     *********************************/
    gridContainer.cubeportfolio({
        defaultFilter: '*',
        animationType: 'fadeOutTop',
        gapHorizontal: 0,
        gapVertical: 0,
        gridAdjustment: 'responsive',
        mediaQueries: [{
            width: 1600,
            cols: 5
        },{
            width: 1200,
            cols: 5
        }, {
            width: 800,
            cols: 3
        }, {
            width: 500,
            cols: 2
        }, {
            width: 320,
            cols: 1
        }],
        caption: 'zoom',
        displayType: 'lazyLoading',
        displayTypeSpeed: 100,

        // lightbox
        lightboxDelegate: '.cbp-lightbox',
        lightboxGallery: true,
        lightboxTitleSrc: 'data-title',
        lightboxCounter: '<div class="cbp-popup-lightbox-counter">{{current}} of {{total}}</div>',

        // singlePage popup
        singlePageDelegate: '.cbp-singlePage',
        singlePageDeeplinking: true,
        singlePageStickyNavigation: true,
        singlePageCounter: false,
        singlePageCallback: function(url, element) {
            var t = this;
            $.ajax({
                url: url,
                type: 'GET',
                dataType: 'html',
                timeout: 5000
            })
            .done(function(result) {
                t.updateSinglePage(result);
                $(".container").fitVids();
            })
            .fail(function() {
                t.updateSinglePage("Error! Please refresh the page!");
            });
        },

        // singlePageInline
        singlePageInlineDelegate: '.cbp-singlePageInline',
        singlePageInlinePosition: 'below',
        singlePageInlineInFocus: true,
        singlePageInlineCallback: function(url, element) {
            var t = this;
            $.ajax({
                url: url,
                type: 'GET',
                dataType: 'html',
                timeout: 5000
            })
            .done(function(result) {              
                t.updateSinglePage(result);
                $(".container").fitVids();
            })
            .fail(function() {
                t.updateSinglePage("Error! Please refresh the page!");
            });
        }
    });


    /*********************************
        add listener for filters
     *********************************/
    if (filtersContainer.hasClass('cbp-l-filters-dropdown')) {
        wrap = filtersContainer.find('.cbp-l-filters-dropdownWrap');

        wrap.on({
            'mouseover.cbp': function() {
                wrap.addClass('cbp-l-filters-dropdownWrap-open');
            },
            'mouseleave.cbp': function() {
                wrap.removeClass('cbp-l-filters-dropdownWrap-open');
            }
        });

        filtersCallback = function(me) {
            wrap.find('.cbp-filter-item').removeClass('cbp-filter-item-active');
            wrap.find('.cbp-l-filters-dropdownHeader').text(me.text());
            me.addClass('cbp-filter-item-active');
            wrap.trigger('mouseleave.cbp');
        };
    } else {
        filtersCallback = function(me) {
            me.addClass('cbp-filter-item-active').siblings().removeClass('cbp-filter-item-active');
        };
    }

    filtersContainer.on('click.cbp', '.cbp-filter-item', function() {
        var me = $(this);

        if (me.hasClass('cbp-filter-item-active')) {
            return;
        }

        // get cubeportfolio data and check if is still animating (reposition) the items.
        if (!$.data(gridContainer[0], 'cubeportfolio').isAnimating) {
            filtersCallback.call(null, me);
        }

        // filter the items
        gridContainer.cubeportfolio('filter', me.data('filter'), function() {});
    });

})(jQuery, window, document);