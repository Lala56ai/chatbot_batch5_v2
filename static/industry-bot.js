var loadingMsgIndex,
    botui = new BotUI('industry-bot'),
    country = '',
    local = '',
	gender = '',
	date = '',
	isection = '',
	eType = '',
	risk = '',
    API = '/predict',
    coutryLookup = [{text: 'Country_01',value: 'Country_01'}, {text: 'Country_02',value: 'Country_02'},{text: 'Country_03',  value: 'Country_03'}],
	localLookup1 = [{text: 'Local_01',value: 'Local_01'}, {text: 'Local_03',value: 'Local_03'}, {text: 'Local_04',value: 'Local_04'}, {text: 'Local_06',value: 'Local_06'}, {text: 'Local_11',value: 'Local_11'}],
	localLookup2 = [{text: 'Local_02',value: 'Local_02'}, {text: 'Local_05',value: 'Local_05'}, {text: 'Local_07',value: 'Local_07'}, {text: 'Local_08',value: 'Local_08'}, {text: 'Local_09',value: 'Local_09'}, {text: 'Local_12',value: 'Local_12'}],
	localLookup3 = [{text: 'Local_10',value: 'Local_10'}],
	industryLookup = [{text: 'Mining',value: 'Mining'},{text: 'Metals',value: 'Metals'},{text: 'others',value: 'others'}],
	genderLookup = [{text: 'Male',value: 'Male'},{text: 'Female',value: 'Female'}],
	eTypeLookup = [{text: 'Employee',value: 'Employee'},{text: 'Third Party',value: 'Third Party'},{text: 'Third Party (Remote)',value: 'Third Party (Remote)'}],
	
	riskLookup = [{text: 'Not applicable',value: 'Not applicable'},{text: 'Pressed',value: 'Pressed'},{text: 'Pressurized Systems',value: 'Pressurized Systems'},{text: 'Manual Tools',value: 'Manual Tools'},{text: 'Others',value: 'Others'},{text: 'Fall prevention (same level)',value: 'Fall prevention (same level)'},{text: 'Chemical substances',value: 'Chemical substances'},{text: 'Liquid Metal',value: 'Liquid Metal'},{text: 'Electrical installation',value: 'Electrical installation'},{text: 'Confined space',value: 'Confined space'},{text: 'Pressurized Systems / Chemical Substances',value: 'Pressurized Systems / Chemical Substances'},{text: 'Blocking and isolation of energies',value: 'Blocking and isolation of energies'},{text: 'Suspended Loads',value: 'Suspended Loads'},{text: 'Poll',value: 'Poll'},{text: 'Cut',value: 'Cut'},{text: 'Fall',value: 'Fall'},{text: 'Bees',value: 'Bees'},{text: 'Fall prevention',value: 'Fall prevention'},{text: 'Traffic',value: 'Traffic'},{text: 'Projection',value: 'Projection'},{text: 'Venomous Animals',value: 'Venomous Animals'},{text: 'Plates',value: 'Plates'},{text: 'Projection/Burning',value: 'Projection/Burning'},{text: 'remains of choco',value: 'remains of choco'},{text: 'Vehicles and Mobile Equipment',value: 'Vehicles and Mobile Equipment'},{text: 'Projection/Choco',value: 'Projection/Choco'},{text: 'Machine Protection',value: 'Machine Protection'},{text: 'Power lock',value: 'Power lock'},{text: 'Burn',value: 'Burn'},{text: 'Projection/Manual Tools',value: 'Projection/Manual Tools'},{text: 'Individual protection equipment',value: 'Individual protection equipment'},{text: 'Electrical Shock',value: 'Electrical Shock'},{text: 'Projection of fragments',value: 'Projection of fragments'}];
	
var askCountry = function () {
botui.message.removeAll();
botui.message
  .bot('Select your country :')
  .then(function () {
    return botui.action.button({
      delay: 500,
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
}

var askLocal = function () {
	
if(country == 'Country_01'){
	botui.message
  .bot('Select your Local :')
  .then(function () {
    return botui.action.button({
      delay: 500,
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
else if(country == 'Country_02'){
	botui.message
  .bot('Select your Local :')
  .then(function () {
    return botui.action.button({
      delay: 500,
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
      delay: 500,
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
      delay: 500,
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
      delay: 500,
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
      delay: 500,
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
	value: 'Pressed', // Selected value or selected object. Example: {value: "TR", text : "TÃ¼rkÃ§e" }
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
      delay: 500,
      action: {
		sub_type: 'date',
        value: '2016-01-01',
        placeholder: '2016-01-01'
      }
    })
}).then(function (res) {   
    
	date = res.value;
	console.log("Date :"+ date); 
	askKey();
});

}



function askKey() {
  botui.message
  .bot({
    delay: 500,
    content: 'Provide the description to find the Accident and Potential Accident level:'
  })
  .then(function () {
    return botui.action.text({
      delay: 500,
      action: {
		size: 50,
        value: '',
        placeholder: ''
      }
    })
  }).then(function (res) {
    loadingMsgIndex = botui.message.bot({
      delay: 5000,
      loading: true
    }).then(function (index) {
      loadingMsgIndex = index;
      sendXHR(res.value,showAccidentLevel)
    });
  });
}

var continuefunc = function(){
	botui.message
  .bot('Do you want to Continue ?')
  .then(function () {
    return botui.action.button({
      delay: 10,
      addMessage: false, // so we could the address in message instead if 'Existing Address'
      action: [{
        text: 'YES',
        value: 'yes'
      }, {
        text: 'NO',
        value: 'no'
      }]
    })
}).then(function (res) {
	botui.message.human({
      delay: 10,
      content: res.text
    });
  if(res.value == 'yes') {       
    askKey();
  } else {
     botui.message
    .bot({
      delay: 500,
      content: 'Thank you for chatting with us !!'
    });
  }
});
}



function sendXHR(key,showAccidentLevel) {
    
var text = "{\"accident_deatils\": {\"Country\": \""+country+"\",\"Local\": \""+local+"\",\"Industry_Sector\": \""+isection+"\",\"Gender\": \""+gender+"\",\"Employee_Type\": \""+eType+"\",\"Critical_Risk\": \""+risk+"\",\"Date\": \""+date+"\",\"Description\":\""+key+"\"}}"

  
  $.ajax({
            url: API,
            type: 'Post',
            dataType: 'json',
            contentType: 'application/json',
			data: text,
            success: function (data) {                
				showAccidentLevel(data['Accident pedict'],data['Potential Accident pedict'])
            },
			error: function (err) {                
				showErrorMessage(err)
            }
			
        });
}


function showAccidentLevel(lvl1,lvl2) {
  botui.message
  .update(loadingMsgIndex, {
    content: 'Accident Level :' + (lvl1 || "Not found") +' Potential Accident Level :' + (lvl2 || "Not found")
  })
  .then(continuefunc); // ask again for repo. Keep in loop.
}

function showErrorMessage(err) {
  botui.message
  .update(loadingMsgIndex, {
    content: 'Something wrong with the request'
  })
  .then(continuefunc); // ask again for repo. Keep in loop.
}

$( ".cancel" ).on( "click", function() {	
  document.getElementById("myForm").style.display = "none";
    country = '';
    local = '';
	gender = '';
	date = '';
	isection = '';
	eType = '';
	risk = '';
	botui.message.removeAll();
});
$( ".open-button" ).on( "click", function() {
    country = '';
    local = '';
	gender = '';
	date = '';
	isection = '';
	eType = '';
	risk = '';	
  askCountry();
});
