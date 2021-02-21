var loadingMsgIndex,
    botui = new BotUI('industry-bot'),
    country = '1',
    local = '1',
	gender = '1',
	date = '2012-04-23',
	isection = '1',
	eType = '1',
	risk = '0',
    API = '/predict',
    coutryLookup = [{text: 'Country_01',value: '1'}, {text: 'Country_02',value: '2'},{text: 'Country_03',  value: '3'}],
	localLookup1 = [{text: 'Local_01',value: '1'}, {text: 'Local_03',value: '2'}, {text: 'Local_04',value: '4'}, {text: 'Local_06',value: '6'}, {text: 'Local_11',value: '11'}],
	localLookup2 = [{text: 'Local_02',value: '2'}, {text: 'Local_05',value: '5'}, {text: 'Local_07',value: '7'}, {text: 'Local_08',value: '8'}, {text: 'Local_09',value: '9'}, {text: 'Local_12',value: '12'}],
	localLookup3 = [{text: 'Local_10',value: '10'}],
	industryLookup = [{text: 'Mining',value: '1'},{text: 'Metals',value: '2'},{text: 'others',value: '3'}],
	genderLookup = [{text: 'Male',value: '1'},{text: 'Female',value: '2'}],
	eTypeLookup = [{text: 'Employee',value: '1'},{text: 'Third Party',value: '2'},{text: 'Third Party (Remote)',value: '3'}],
	riskLookup = [{text: 'Not Applicable',value: '0'},{text: 'Bees',value: '1'},{text: 'Blocking and isolation of energies)',value: '2'}];
	

botui.message
  .bot('Select your country :')
  .then(function () {
    return botui.action.button({
      delay: 1000,
      addMessage: false, 
      action: coutryLookup
    })
}).then(function (res) {
   
    botui.message.human({
      delay: 500,
      content: res.text
    });
	country = res.value;
	console.log("Country :"+ country); 
	askLocal();
  
});

var askLocal = function () {
	
if(country == '1'){
	botui.message
  .bot('Select your Local :')
  .then(function () {
    return botui.action.button({
      delay: 1000,
      addMessage: false, 
      action: localLookup1
    })
}).then(function (res) {
   
    botui.message.human({
      delay: 500,
      content: res.text
    });
	local = res.value;
	console.log("Local :"+ local); 
	askIndustrySection();
});
	}
else if(country == '2'){
	botui.message
  .bot('Select your Local :')
  .then(function () {
    return botui.action.button({
      delay: 1000,
      addMessage: false, 
      action: localLookup2
    })
}).then(function (res) {
   
    botui.message.human({
      delay: 500,
      content: res.text
    });
	local = res.value;
	console.log("Local :"+ local); 
	askIndustrySection();
});
}

else{
	botui.message
  .bot('Select your Local :')
  .then(function () {
    return botui.action.button({
      delay: 1000,
      addMessage: false, 
      action: localLookup3
    })
}).then(function (res) {
   
    botui.message.human({
      delay: 500,
      content: res.text
    });
	local = res.value;
	console.log("Local :"+ local);  
	askIndustrySection();
  
});
}

}

var askIndustrySection = function () {
	botui.message
  .bot('Select your Industry Section :')
  .then(function () {
    return botui.action.button({
      delay: 1000,
      addMessage: false, 
      action: industryLookup
    })
}).then(function (res) {
   
    botui.message.human({
      delay: 500,
      content: res.text
    });
	isection = res.value;
	console.log("Industry Section :"+ isection); 
	askEType();
});

}

var askEType = function () {
	botui.message
  .bot('Select your Employee Type :')
  .then(function () {
    return botui.action.button({
      delay: 1000,
      addMessage: false, 
      action: eTypeLookup
    })
}).then(function (res) {
   
    botui.message.human({
      delay: 500,
      content: res.text
    });
	eType = res.value;
	console.log("Employee Type :"+ eType); 
	askGender();
	
});

}



var askGender = function () {
	botui.message
  .bot('Select your Gender :')
  .then(function () {
    return botui.action.button({
      delay: 1000,
      addMessage: false, 
      action: genderLookup
    })
}).then(function (res) {
   
    botui.message.human({
      delay: 500,
      content: res.text
    });
	gender = res.value;
	console.log("Gender :"+ gender); 
	//init();
	askCriticalRisk();
});
}

var askCriticalRisk = function () {
	botui.message
  .bot('Select the Risk :')
  .then(function () {
    return botui.action.select({
	action: {
	placeholder : "Select Risk",
	value: '1', // Selected value or selected object. Example: {value: "TR", text : "Türkçe" }
	searchselect : true, // Default: true, false for standart dropdown
	label : 'text', // dropdown label variable
	options : riskLookup,
	button: {
	icon: 'check',
	label: 'OK'
	}
	}
	}).then(function (res) { 
	risk = res.value;
	console.log("Risk :"+ risk); 
	askDate();
});
  });
  
}

var askDate = function () {
	botui.message
  .bot('Select the Date :')
  .then(function () {
    return botui.action.text({
      delay: 1000,
      action: {
        value: '2012-04-23',
        placeholder: 'YYYY-MM-DD'
      }
    })
}).then(function (res) {   
    botui.message.human({
      delay: 500,
      content: res.value
    });
	date = res.value;
	console.log("Date :"+ date); 
	init();
});

}

function sendXHR(repo, cb) {
  var xhr = new XMLHttpRequest();
  var self = this;
  xhr.open('GET', API + repo);
  xhr.onload = function () {
    var res = JSON.parse(xhr.responseText)
    cb(res.stargazers_count);
  }
  xhr.send();
}

function init() {
  botui.message
  .bot({
    delay: 1000,
    content: 'Enter the keyword to find the accident level:'
  })
  .then(function () {
    return botui.action.text({
      delay: 1000,
      action: {
        value: 'pulley',
        placeholder: 'pulley'
      }
    })
  }).then(function (res) {
    loadingMsgIndex = botui.message.bot({
      delay: 200,
      loading: true
    }).then(function (index) {
      loadingMsgIndex = index;
      sendXHR1(res.value)
    });
  });
}

function showStars(stars) {
  botui.message
  .update(loadingMsgIndex, {
    content: 'it has !(star) ' + (stars || "0") + ' stars.'
  })
  .then(init); // ask again for repo. Keep in loop.
}


function sendXHR1(key) {

		
    var text = '{ "accident_deatils" : [' +
'{ "Country":'+country+' , "Local":'+local+', "Industry_Sector":'+isection+', "Gender":'+gender+', "Employee_Type":'+eType+', "Critical_Risk":'+risk+' , "Date":"'+date+'" , "Description":"'+key+'" }'+ ']}';
var obj = JSON.parse(text);
	
  var xhr = new XMLHttpRequest();
  xhr.open('POST', API);
  xhr.onload = function () {
    var res = JSON.parse(xhr.responseText)
  }
  xhr.send(text);
}

/*
function sendXHR1(key) {
  var alvl = LoadJson(key);
  showAccidentLevel(alvl);
}
*/

function LoadJson(key){
	var accident_deatils = {
          "Country": country,
		  "Local": local,
		  "Industry_Sector":isection,
		  "Gender":gender,
		  "Employee_Type":eType,
		  "Critical_Risk":risk,
		  "Date" :date,
		  "Description":key
       };
	//var url = "url?data=" + encodeURIComponent(JSON.stringify({"country": country, "local": local, "gender": gender, "isection": isection, "eType": eType,"":}));
	//console.log(url);
	var json = null;
    $.ajax({
        'async': false,
        'global': false,
        'url': API,
		'body' : accident_deatils,
        'dataType': "json",
        'success': function (data) {
            json = data;
        }
		
    });
return(json[key]);
console.log(json[key]);
}

function showAccidentLevel(lvl) {
  botui.message
  .update(loadingMsgIndex, {
    content: 'Accident Level :' + (lvl || "Not found")  
  })
  .then(init); // ask again for repo. Keep in loop.
}



