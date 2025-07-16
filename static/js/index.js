window.HELP_IMPROVE_VIDEOJS = false;


var INTERP_BASE = "./static/interpolation/interactive";
var NUM_INTERP_FRAMES = 24;
var old_img_t = 6;
var canv = document.createElement("canvas");
const interp_images = Object.create(null);
// var interp_images = [];
const dists = [22, 0, 37, 135, 155, 170];
var disparity_image;
var pixelData;
var image;
var current_dist = 170;
var demo_guide;

function preloadInterpolationImages() {

  for(var j = 0; j < dists.length; j++){
  var dist = dists[j];
  interp_images[dist] = []
    for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
      var path = INTERP_BASE + '/' + "model_output_aperture_" + i + "_focus_" + dist + ".jpg";
      interp_images[dist][i] = new Image();
      interp_images[dist][i].src = path;
    }
    demo_guide = new Image()
    demo_guide.src = INTERP_BASE + '/demo_guide.jpg'
}
  disparity_image = new Image();
  disparity_image.src = "./static/interpolation/copy/rounded.png";
}

function initInterpolationImage() {
  image = demo_guide;
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  // console.log("fake image size " + image.width + " " + image.height + " wrapper width maybe " + $('#interpolation-image-wrapper').width);
  image.onmousedown = GetCoordinates;
  
  $('#interpolation-image-wrapper').empty().append(image);
}

function setInterpolationImage(i) {
  console.log("old image " + old_img_t + "current dist " + current_dist );
  old_img_t = i;
  image = interp_images[current_dist][i];

  canv.width = disparity_image.width;
  canv.height = disparity_image.height;
  var ctx = canv.getContext("2d");
  ctx.drawImage(disparity_image, 0, 0);
  var c = canv.getContext('2d');
  var p = c.getImageData(10, 10, 1, 1).data;
  // console.log("canv width" + canv.width);

  var hex = "RGB = " + p[0]+", "+p[1]+", "+p[2];
  console.log("color " + hex);
  if(pixelData != null){
    console.log("pixel data " + pixelData[i]);
  }
  

  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  // console.log("fake image size " + image.width + " " + image.height + " wrapper width maybe " + $('#interpolation-image-wrapper').width);
  image.onmousedown = GetCoordinates;
  
  $('#interpolation-image-wrapper').empty().append(image);
}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    

    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value); //console.log(this.value); console.log($('#interpolation-slider')); console.log(this);
    });

    // setInterpolationImage(14);
    initInterpolationImage();
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);
    $('#interpolation-slider').prop('value', old_img_t);

    bulmaSlider.attach();

})

const img = new Image();

// When the image is loaded, create a canvas element and draw the image on it
img.onload = function() {
  const canvas = document.createElement('canvas');
  canvas.width = img.width;
  canvas.height = img.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, img.width, img.height);
  console.log("Image width " + img.width)

  // Get the pixel data from the canvas
  const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
  pixelData = imageData.data;

  // Get the color of the pixel at position (x, y)
  const x = 10;
  const y = 10;
  const index = (y * canvas.width + x) * 4; // Multiply by 4 because each pixel is represented by 4 values (R,G,B,A)
  const red = pixelData[index];
  const green = pixelData[index + 1];
  const blue = pixelData[index + 2];
  const alpha = pixelData[index + 3];

  // Print the color values to the console
  console.log(`Pixel color at position (${x}, ${y}): rgb(${red}, ${green}, ${blue}), alpha: ${alpha}`);
}

// Set the source of the image
img.src = './static/interpolation/copy/rounded.png';

function FindPosition(oElement)
{
  if(typeof( oElement.offsetParent ) != "undefined")
  {
    for(var posX = 0, posY = 0; oElement; oElement = oElement.offsetParent)
    {
      posX += oElement.offsetLeft;
      posY += oElement.offsetTop;
    }
      return [ posX, posY ];
    }
    else
    {
      return [ oElement.x, oElement.y ];
    }
}
function GetCoordinates(e)
{
  var PosX = 0;
  var PosY = 0;
  var ImgPos;
  ImgPos = FindPosition(image);
  if (!e) var e = window.event;
  if (e.pageX || e.pageY)
  {
    PosX = e.pageX;
    PosY = e.pageY;
  }
  else if (e.clientX || e.clientY)
    {
      PosX = e.clientX + document.body.scrollLeft
        + document.documentElement.scrollLeft;
      PosY = e.clientY + document.body.scrollTop
        + document.documentElement.scrollTop;
    }
  PosX = PosX - ImgPos[0];
  PosY = PosY - ImgPos[1];

  console.log("window width " +  window.width);

  var element = document.querySelector('#interpolation-image-wrapper');
  let width = element.offsetWidth;
  let height = element.offsetHeight;

  var scaledPosY = Math.abs(Math.round(2024 / height * PosY));
  var scaledPosX = Math.abs(Math.round(2688 / width * PosX));

  var pixel_index = (scaledPosY * 2688 + scaledPosX) * 4; // Multiply by 4 because each pixel is represented by 4 values (R,G,B,A)
  red_val = pixelData[pixel_index + 0];
  green_val = pixelData[pixel_index + 1];
  blue_val = pixelData[pixel_index + 2];
  current_dist = blue_val;
  setInterpolationImage(old_img_t);
  // document.getElementById("x").innerHTML = PosX;
  // document.getElementById("y").innerHTML = PosY;
  // console.log(width, height);
  // console.log(scaledPosY, scaledPosX);
  // console.log("RGB " + red_val + " " + green_val + " " + blue_val);
  // console.log("touch x " + PosX + " y " + PosY);
}


function moveDivisor() { 
  var divisor = document.getElementById("divisor"),
  slider = document.getElementById("compare_slider");
	divisor.style.width = slider.value+"%";
}

function changedata(parameter){
  if(parameter==0){
      document.getElementById('kitchen_deblurring').style.display = 'block';
      document.getElementById('book_deblurring').style.display = 'none';
      document.getElementById('example1_deblurring_btn').className = 'btn active';
      document.getElementById('example2_deblurring_btn').className = 'btn';
  }
  else if(parameter==1){
      document.getElementById('book_deblurring').style.display = 'block';
      document.getElementById('kitchen_deblurring').style.display = 'none';
      document.getElementById('example1_deblurring_btn').className = 'btn';
      document.getElementById('example2_deblurring_btn').className = 'btn active';
  }
  this.className += " active";
}

function changeexample(image_to_show, image_to_hide, btn_to_active, btn_to_normal){
      document.getElementById(image_to_show).style.display = 'block';
      document.getElementById(image_to_hide).style.display = 'none';
      document.getElementById(btn_to_active).className = 'btn active';
      document.getElementById(btn_to_normal).className = 'btn';
}

function toggleimage(image_to_show, image_to_hide, btn_to_active, btn_to_normal){
  document.getElementById(image_to_show).style.display = 'block';
  document.getElementById(image_to_hide).style.display = 'none';
  document.getElementById(btn_to_active).className = 'btn active';
  document.getElementById(btn_to_normal).className = 'btn';
}


$(function(){
  $(".compare").twentytwenty({
    default_offset_pct: 0.5, // How much of the before image is visible when the page loads
    orientation: 'horizontal', // Orientation of the before and after images ('horizontal' or 'vertical')
    before_label: 'Input', // Set a custom before label
    after_label: 'Output', // Set a custom after label
    no_overlay: true, //Do not show the overlay with before and after
    move_slider_on_hover: true, // Move slider on mouse hover?
    move_with_handle_only: true, // Allow a user to swipe anywhere on the image to control slider movement. 
    click_to_move: true // Allow a user to click (or tap) anywhere on the image to move the slider to that location.
  });
});

$('.compare').imagesLoaded(function() {
  $(".compare").twentytwenty({
    default_offset_pct: 0.5, // How much of the before image is visible when the page loads
    orientation: 'horizontal', // Orientation of the before and after images ('horizontal' or 'vertical')
    before_label: 'Input', // Set a custom before label
    after_label: 'Output', // Set a custom after label
    no_overlay: true, //Do not show the overlay with before and after
    move_slider_on_hover: true, // Move slider on mouse hover?
    move_with_handle_only: true, // Allow a user to swipe anywhere on the image to control slider movement. 
    click_to_move: true // Allow a user to click (or tap) anywhere on the image to move the slider to that location.
  });
  console.log("images loaded");
});

setTimeout(function() {
  // $(".compare").twentytwenty({
  //   default_offset_pct: 0.5, // How much of the before image is visible when the page loads
  //   orientation: 'horizontal', // Orientation of the before and after images ('horizontal' or 'vertical')
  //   before_label: 'Input', // Set a custom before label
  //   after_label: 'Output', // Set a custom after label
  //   no_overlay: true, //Do not show the overlay with before and after
  //   move_slider_on_hover: true, // Move slider on mouse hover?
  //   move_with_handle_only: true, // Allow a user to swipe anywhere on the image to control slider movement. 
  //   click_to_move: true // Allow a user to click (or tap) anywhere on the image to move the slider to that location.
  // });
  window.dispatchEvent(new Event('resize')); 
  console.log("loaded after tiemout");
  document.getElementById('book_deblurring').style.display = 'none';
  document.getElementById('microkitchen_refocusing').style.display = 'none';
  }, 2000);

// $(window).load(function() {
//   $(window).trigger("resize.twentytwenty");
// });
