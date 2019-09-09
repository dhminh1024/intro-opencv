CAPTURE_IMG_WIDTH = 640
CAPTURE_IMG_HEIGHT = 480

// Show the selected image to the UI before uploading
function readURL(input, id) {
  if (input.files && input.files[0]) {
    var reader = new FileReader();
    
    reader.onload = function(e) {
      $(id).attr('src', e.target.result).css({'width': CAPTURE_IMG_WIDTH, 'height': CAPTURE_IMG_HEIGHT});
    }
    
    reader.readAsDataURL(input.files[0]);
  }
}

mode = 'canvas'

// TABS HANDLING
$('.tabs').on('click', '.button', function() {
  let target = $(this).attr('data-target');
  console.log(target)
  $('.tab-content').addClass('hidden')
  $('.tab-' + target).removeClass('hidden')
  $('.tabs .button').removeClass('is-info').addClass('is-outlined')
  $(this).addClass('is-info').removeClass('is-outlined')
  if(target == 'camera') {
    initialize_webcam();
    $('.btn-clear-canvas').addClass('hidden');
  } else if (target == 'canvas') {
    initialize_canvas();
    $('.btn-clear-canvas').removeClass('hidden');
  }
  mode = target
})

function initialize_webcam() {
  // HTML5 WEBCAM
  Webcam.set({
    width: CAPTURE_IMG_WIDTH,
    height: CAPTURE_IMG_HEIGHT,
    image_format: 'jpeg',
    jpeg_quality: 90
  });
  Webcam.attach( '#my-camera' );
  $('.camera-guide, .camera-wrapper').css({width: CAPTURE_IMG_WIDTH, height: CAPTURE_IMG_HEIGHT})
}

function initialize_canvas() {
  var canvas = document.querySelector("#canvas");
	var context = canvas.getContext("2d");
	canvas.width = 640;
	canvas.height = 280;

	var Mouse = {x:0, y:0};
	var lastMouse = {x:0, y:0};
	context.fillStyle = "white";
	// context.fillRect(0, 0, canvas.width, canvas.height);
	context.color = "white";
	context.lineWidth = 10;
  context.lineJoin = context.lineCap = 'round';

	canvas.addEventListener("mousemove", function(e) {
		lastMouse.x = Mouse.x;
		lastMouse.y = Mouse.y;

		Mouse.x = e.pageX - this.offsetLeft-30;
    Mouse.y = e.pageY - this.offsetTop-90;
	}, false);

	canvas.addEventListener("mousedown", function(e) {
		canvas.addEventListener("mousemove", onPaint, false);
	}, false);

	canvas.addEventListener("mouseup", function() {
		canvas.removeEventListener("mousemove", onPaint, false);
	}, false);

	var onPaint = function() {	
		context.lineWidth = context.lineWidth;
		context.lineJoin = "round";
		context.lineCap = "round";
		context.strokeStyle = context.color;
	
		context.beginPath();
		context.moveTo(lastMouse.x, lastMouse.y);
		context.lineTo(Mouse.x,Mouse.y );
		context.closePath();
		context.stroke();
	};
}

function isCanvasBlank(canvas) {
  const context = canvas.getContext('2d');
  const pixelBuffer = new Uint32Array(
    context.getImageData(0, 0, canvas.width, canvas.height).data.buffer
  );
  return !pixelBuffer.some(color => color !== 0);
}

function clear_canvas() {

  const context = canvas.getContext('2d');
  context.clearRect(0, 0, canvas.width, canvas.height);
  // context.fillStyle="white";
  // context.fillRect(0,0,canvas.width,canvas.height);
}

$('.btn-clear-canvas').on('click', function () {
  $('.results').addClass('hidden');
  clear_canvas()
});

// CALCULATE
let form_capture = document.getElementById('form-capture-image')
$('.btn-capture-image').on('click', function(e) {
  e.preventDefault();

  if(mode == 'camera') {
    $(this).addClass('is-loading');
    Webcam.snap(function(data_uri) {
      // display results in page
      let json_data = {'data-uri': data_uri, 'mode': mode }
      let camera = $('#my-camera')
      // $('#my-camera').addClass('hidden');
      // $('.taken-photo').attr('src', data_uri).removeClass('hidden');
      handleAjax(json_data)
    });
  } else if (mode == 'canvas') {
    let canvasObj = document.getElementById("canvas");
    const blank_canvas = isCanvasBlank(canvasObj)
    // console.log(blank_canvas)
    if (blank_canvas) {
      $('.results').removeClass('hidden').html('<h3>Please write your equation!</h3>')
    } else {
      $(this).addClass('is-loading');
      let img = canvasObj.toDataURL('image/jpeg');
      let json_data = {'data-uri': img, 'mode': mode }
      handleAjax(json_data)
    }
  }
 
});

function handleAjax(json_data) {
  $.ajax({
    type: 'POST',
    url: '/upload/',
    processData: false,
    contentType: 'application/json; charset=utf-8',
    dataType: 'json',
    data: JSON.stringify(json_data),
    success: function(data) {
      $('#my-camera').removeClass('hidden');
      $('.results').removeClass('hidden')
      if (data['status'] == 0) { 
        $('.results').html('<h3>We cannot detect your image, please try again.</h3>')
      } else {
        $('.results').html("<h3>" + data['equation'] + "</h3><img src='data:image/jpeg;base64," + data['image'] + "' alt='' />");
      }
      
      // $('.taken-photo').attr('src', data_uri).addClass('hidden');
      $('.btn-capture-image').removeClass('is-loading')
    }
  });
}

$(document).ready(function() {
  // initialize_webcam();
  initialize_canvas()
});

